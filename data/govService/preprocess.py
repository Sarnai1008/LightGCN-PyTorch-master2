import pandas as pd
import numpy as np

citizen_df = pd.read_csv('citizen.csv')
service_df = pd.read_csv('service.csv')
request_df = pd.read_csv('request.csv')

request_df.rename(columns={"createc_date": "created_date"}, inplace=True)
request_df["created_date"] = pd.to_datetime(request_df["created_date"])

all_users = set(citizen_df['userid']).union(set(request_df['userid']))
all_services = set(service_df['serviceid']).union(set(request_df['serviceid']))

user_map = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
service_map = {service_id: idx for idx, service_id in enumerate(sorted(all_services))}

with open('user_list.txt', 'w') as f:
    for user, idx in user_map.items():
        f.write(f"{user} {idx}\n")

with open('item_list.txt', 'w') as f:
    for service, idx in service_map.items():
        f.write(f"{service} {idx}\n")

# Apply mappings to request data
request_df['userid'] = request_df['userid'].map(user_map)
request_df['serviceid'] = request_df['serviceid'].map(service_map)

# Sort by timestamp (oldest to newest)
request_df = request_df.sort_values(by='created_date')

# Train-test split by timestamp (80% train, 20% test)
split_idx = int(len(request_df) * 0.8)
train_df = request_df.iloc[:split_idx]
test_df = request_df.iloc[split_idx:]

# Ensure every user appears in train set
missing_users = set(test_df['userid']) - set(train_df['userid'])
if missing_users:
    extra_train_data = test_df[test_df['userid'].isin(missing_users)]
    train_df = pd.concat([train_df, extra_train_data])
    test_df = test_df[~test_df['userid'].isin(missing_users)]

# Save train and test data in required format
def save_format(filename, data):
    grouped = data.groupby('userid')['serviceid'].apply(lambda x: ' '.join(map(str, x)))
    with open(filename, 'w') as f:
        for user, items in grouped.items():
            f.write(f"{user} {items}\n")  # No quotes, proper spacing

save_format('train.txt', train_df)
save_format('test.txt', test_df)

print("✅ Preprocessing complete! Files saved: user_list.txt, item_list.txt, train.txt, test.txt")
