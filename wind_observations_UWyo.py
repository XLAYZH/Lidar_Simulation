import os
import requests
import pandas as pd
from datetime import datetime, timedelta#爬取怀俄明探空数据

# ====== 配置参数（只需修改这里）======
station_id = "54511"
start_date = "2024-01-01"
end_date = "2024-12-31"
save_base_path = "I:\舒英杰\气象数据\探空数据"  # ← 自定义保存目录
# ====================================

def fetch_text_data(station_id, dt_str):
    url = "https://weather.uwyo.edu/wsgi/sounding"
    params = {
        "src": "BUFR",
        "datetime": dt_str,
        "id": station_id,
        "type": "TEXT:LIST"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        if "Can't" in response.text or "not available" in response.text:
            print(f"[无数据] {dt_str}")
            return None
        return response.text
    except Exception as e:
        print(f"[请求失败] {dt_str} - {e}")
        return None

def parse_text_to_csv(text):
    lines = text.strip().splitlines()
    data = []
    columns = []
    for line in lines:
        if line.startswith("-----------------------------------------------------------------------------"):
            continue
        elif "PRES" in line and "TEMP" in line:
            columns = line.split()
        elif columns:
            parts = line.split()
            if len(parts) == len(columns):
                data.append(parts)
    if data and columns:
        return pd.DataFrame(data, columns=columns)
    else:
        return None

def save_csv(df, folder, date_str, hour_str):
    os.makedirs(folder, exist_ok=True)
    filename = f"{date_str}_{hour_str}.csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"[已保存] {filepath}")

def main():
    date = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while date <= end:
        date_str = date.strftime("%Y-%m-%d")
        for hour_str, folder_name in [("00:00:00", "00Z"), ("12:00:00", "12Z")]:
            full_dt = f"{date_str} {hour_str}"
            print(f"正在抓取：{full_dt}")
            text = fetch_text_data(station_id, full_dt)
            if text:
                df = parse_text_to_csv(text)
                if df is not None:
                    save_folder = os.path.join(save_base_path, folder_name)
                    save_csv(df, save_folder, date_str, hour_str[:2])
        date += timedelta(days=1)

if __name__ == "__main__":
    main()
