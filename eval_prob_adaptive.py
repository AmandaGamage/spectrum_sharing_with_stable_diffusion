import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
#from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from preprocessing import split_data
from torchvision import transforms
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"


DATASET_ROOT="E:\\Msc\\Lab\\spectrum_sharing_data\\one_channel_all_labeled_data"
prompt_path = "E:\Msc\Lab\spectrum_sharing_with_stable_diffusion\prompts\spectrogram_prompts_one_channel.csv"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def get_target_dataset(args):
    image_transform = torch_transforms.Compose([
        torch_transforms.Resize(args["img_size"], interpolation=InterpolationMode.BICUBIC),
        torch_transforms.CenterCrop(args["img_size"]),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    dataset=split_data.Spectrogram(root_dir=DATASET_ROOT,csv_root=prompt_path,transform=image_transform, target_transform=None)
    return dataset


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)

def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64):
    scheduler_config = get_scheduler_config(args)
    max_n_samples = max(args["n_samples"])
    T = scheduler_config['num_train_timesteps']
    all_noise = torch.randn((max_n_samples * args["n_trials"], 4, latent_size, latent_size), device=latent.device)
    if args["dtype"] == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args["n_samples"], args["to_keep"]):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args["n_trials"])
                noise_idxs.extend(list(range(args["n_trials"] * t_idx, args["n_trials"] * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args["n_trials"])
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args["batch_size"], args["dtype"], args["loss"])
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data

def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors

def main():
    img_size=512
    
    args={
    "interpolation":'bicubic',
    "loss":'l1',
    "img_size":512,
    "dataset":'spectrogram',
    "worker_idx":0,
    "n_workers":1,
    "n_trials":1,
    "version":'2-1',
    "to_keep":[4, 1],
    "n_samples":[4,50],
    "dtype": 'float32',
    "load_stats":True,
    "batch_size":32,
    "loss":'l2'
    }
    assert len(args["to_keep"])==len(args["n_samples"])
    #parser = argparse.ArgumentParser()

    # dataset argshigh
    #parser.add_argument('--dataset', type=str, default='spectrogram',choices=['spectrogram'])
    #parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    #parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    #parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    #parser.add_argument('--batch_size', '-b', type=int, default=32)
    ''' parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    #parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    #parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)'''

    # make run output folder
    name = f"v{args['version']}_{args['n_trials']}trials_"
    name += '_'.join(map(str, str(args["to_keep"]))) + 'keep_'
    name += '_'.join(map(str, str(args["n_samples"]))) + 'samples'
    if args['interpolation'] != 'bicubic':
        name += f'_{interpolation}'
    if args["loss"] == 'l1':
        name += '_l1'
    elif args["loss"] == 'huber':
        name += '_huber'
    if img_size != 512:
        name += f'_{img_size}'

    run_folder = osp.join(LOG_DIR, args["dataset"], name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args['interpolation']]
    #transform = get_transform(interpolation, img_size)
    latent_size = img_size // 8
    target_dataset = get_target_dataset(args)
    prompts_df = pd.read_csv(prompt_path)
    
    true_labels = []
    with open(prompt_path,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_labels.append(row["classname"])


    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # load noise
    '''if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None'''

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    # subset of dataset to evaluate

    idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args["worker_idx"]::args["n_workers"]]
    true_labels=prompts_df.classidx
    accuracy_per_label={label:{"tp": 0, "fp": 0, "total": 0} for label in set(true_labels)}
    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args["load_stats"]:
                data = torch.load(fname)
                if data['pred'] == data['label']:
                    accuracy_per_label[data['label']]["tp"] += 1
                    correct+=1
                else:
                    accuracy_per_label[data['label']]["fp"] += 1
                #correct += int(data['pred'] == data['label'])
                accuracy_per_label[data['label']]["total"] += 1
                total += 1
            continue
        #################################
        image, label =target_dataset[i]
        #image, label = [sublist[0] for sublist in target_dataset[:2]]
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args["dtype"] == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler,  args,latent_size)
        pred = prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            accuracy_per_label[label]["tp"] += 1
            correct += 1
        else:
            accuracy_per_label[label]["fp"] += 1
        total += 1
        accuracy_per_label[label]["total"] += 1
        
    for label, accuracy in accuracy_per_label.items():
        print(f"Accuracy for {label}: {accuracy:.2f}")

if __name__ == '__main__':
    main()
