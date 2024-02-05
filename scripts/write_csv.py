import csv

with open('spectrogram_prompts.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["prompt", "classname", "classidx"]
    
    writer.writerow(field)
    writer.writerow(["CH1 Empty.CH2 Empty", "ch1_empty_ch2_empty", "0"])
    writer.writerow(["CH1 Primary.CH2 Empty", "ch1_primary_ch2_empty", "1"])
    writer.writerow(["CH1 Secondary.CH2 Empty", "ch1_secondary_ch2_empty", "2"])
    writer.writerow(["CH1 Collision.CH2 Empty", "ch1_collision_ch2_empty", "3"])
    writer.writerow(["CH1 Empty.CH2 Primary", "ch1_empty_ch2_primary", "4"])
    writer.writerow(["CH1 Empty.CH2 Secondary", "ch1_empty_ch2_secondary", "5"])
    writer.writerow(["CH1 Empty.CH2 Collision", "ch1_empty_ch2_collision", "6"])
    writer.writerow(["CH1 Primary.CH2 Primary", "ch1_primary_ch2_primary", "7"])
    writer.writerow(["CH1 Primary.CH2 Secondary", "ch1_primary_ch2_secondary", "8"])
    writer.writerow(["CH1 Primary.CH2 Collision", "ch1_primary_ch2_collision", "9"])
    writer.writerow(["CH1 Secondary.CH2 Primary", "ch1_secondary_ch2_primary", "10"])
    writer.writerow(["CH1 Secondary.CH2 Secondary", "ch1_secondary_ch2_secondary", "11"])
    writer.writerow(["CH1 Secondary.CH2 Collision", "ch1_secondary_ch2_collision", "12"])
    writer.writerow(["CH1 Collision.CH2 Primary", "ch1_collision_ch2_primary", "13"])
    writer.writerow(["CH1 Collision.CH2 Secondary", "ch1_collision_ch2_secondary", "14"])
    writer.writerow(["CH1 Collision.CH2 Collision", "ch1_collision_ch2_collision", "15"])