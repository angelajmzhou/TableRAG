import os
import re

folder_path = ["offline_backend/dataset/hybridqa/dev_doc/", "offline_backend/dataset/hybridqa/dev_excel/", "./my_dev.json"]

# List of sports-related terms
sports_terms = [
    "football", "nfl", "soccer", "rugby", "hockey", "basketball", "baseball", "cricket",
    "olympics", "marathon", "decathlon", "athletics", "track", "100m", "200m", "400m", "800m",
    "javelin", "discus", "cross_country", "draft", "season", "tournament", "championship",
    "gymnastics", "swimming", "skiing", "snowboarding", "karate", "judo", "wrestling", "boxing",
    "fencing", "skating", "rowing", "cycling", "triathlon", "pentathlon", "weightlifting",
    "golf", "tennis", "badminton", "volleyball", "handball", "snooker", "motorsport", "grand_prix",
    "nascar", "indycar", "mlb", "nba", "nhl", "mls"
]
pattern = re.compile(r"|".join(sports_terms), re.IGNORECASE)

# Delete non-sports files
for i in range (0,2,1):
    for fname in os.listdir(folder_path[i]):
        if not pattern.search(fname):
            os.remove(os.path.join(folder_path[i], fname))
