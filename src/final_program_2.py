import os
import sys
import argparse
import glob
import time
import csv
import cv2
import numpy as np
from collections import defaultdict, Counter, deque
from ultralytics import YOLO
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


def get_unique_filename(base_name, extension, directory):
    count = 1
    while True:
        filename = os.path.join(directory, f"{base_name}_{count}.{extension}")
        if not os.path.exists(filename):
            return filename
        count += 1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rack_model', help='Path to YOLO rack model file', required=True)
    parser.add_argument('--obj_model', help='Path to YOLO object model file', required=True)
    parser.add_argument('--source', help='Image source: image file, folder, video file, or USB index (e.g., usb0)', required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
    parser.add_argument('--resolution', help='Resolution WxH (e.g., 640x480)', default=None)
    parser.add_argument('--record', help='Record results to demo1.avi', action='store_true')
    return parser.parse_args()

def determine_source_type(img_source, img_ext, vid_ext):
    if os.path.isdir(img_source):
        return 'folder'
    elif os.path.isfile(img_source):
        ext = os.path.splitext(img_source)[1]
        if ext in img_ext:
            return 'image'
        elif ext in vid_ext:
            return 'video'
        else:
            print('Unsupported file extension.'); sys.exit()
    elif 'usb' in img_source:
        return 'usb'
    else:
        print('Invalid source input.'); sys.exit()

def init_recorder(source_type, user_res):
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    resW, resH = map(int, user_res.split('x'))
    record_name = 'demo_full.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))
    return recorder, resW, resH

def prepare_input_source(source_type, img_source, img_ext, user_res):
    if source_type == 'image':
        return [img_source], None
    elif source_type == 'folder':
        return [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext], None
    elif source_type in ['video', 'usb']:
        idx = int(img_source[3:]) if source_type == 'usb' else img_source
        cap = cv2.VideoCapture(idx)
        if user_res:
            resW, resH = map(int, user_res.split('x'))
            cap.set(3, resW)
            cap.set(4, resH)
        return None, cap

def log_summary(output_dir, total_frames, total_objects, total_racks, csv_file, avg_fps,inference_time):
    """
    Logs a summary of the detection run to a text file.

    Parameters:
        output_dir (str): Directory to store the summary log.
        total_frames (int): Number of processed frames.
        total_objects (int): Number of detected objects.
        total_racks (int): Number of racks detected in the last frame.
        csv_file (str): Path to the saved CSV file.
        avg_fps (float): Average frames per second of the detection process.
    """
    summary_text = (
        "\n--- Detection Summary ---\n"
        f"Total Frames Processed: {total_frames}\n"
        f"Total Objects Detected: {total_objects}\n"
        f"Total Racks Detected in Last Frame: {total_racks}\n"
        f"CSV Output File: {csv_file}\n"
        f"Average FPS: {avg_fps:.2f}\n"
        f"Inference Time : {inference_time:.8f} seconds\n"
    )

    print(summary_text)

    log_file = os.path.join(output_dir, 'run_summary.txt')
    with open(log_file, 'w') as f:
        f.write(summary_text)

    print(f"Summary saved to: {log_file}")

def calculate_framewise_mae_rmse(summary_df, ground_truth_ranges, output_dir):
    """
    Calculates per-frame and per-range MAE and RMSE for object detection accuracy.

    Args:
        summary_df (pd.DataFrame): DataFrame containing 'frame_id' and 'total_objects'.
        ground_truth_ranges (list of tuples): Each tuple is (start_frame, end_frame, true_count).
        output_dir (str): Directory to save output CSVs.

    Returns:
        tuple: (per_frame_df, per_range_df)
    """

    # Step 1: Clean frame ID to numeric
    summary_df['frame_num'] = summary_df['frame_id'].str.extract(r'frame_(\d+)').astype(int)
    frame_df = summary_df.drop_duplicates(subset='frame_num')[['frame_num', 'total_objects']].copy()
    frame_df.rename(columns={'total_objects': 'predicted_count'}, inplace=True)

    # Step 2: Map ground truth
    def get_ground_truth(frame_num):
        for start, end, count in ground_truth_ranges:
            if start <= frame_num <= end:
                return count
        return np.nan

    frame_df['ground_truth_count'] = frame_df['frame_num'].apply(get_ground_truth)

    # Step 3: Calculate errors
    frame_df['abs_error'] = (frame_df['predicted_count'] - frame_df['ground_truth_count']).abs()
    frame_df['squared_error'] = (frame_df['predicted_count'] - frame_df['ground_truth_count']) ** 2
    frame_df['MAE'] = frame_df['abs_error']
    frame_df['RMSE'] = frame_df['squared_error']

    # Step 4: Save per-frame metrics
    per_frame_csv = os.path.join(output_dir, 'framewise_accuracy_metrics.csv')
    frame_df.to_csv(per_frame_csv, index=False)
    print(f"✅ Saved per-frame MAE/RMSE to: {per_frame_csv}")

    # Step 5: Aggregate per range
    range_summary = []
    for start, end, true_count in ground_truth_ranges:
        range_df = frame_df[(frame_df['frame_num'] >= start) & (frame_df['frame_num'] <= end)]
        avg_mae = range_df['MAE'].mean()
        avg_rmse = range_df['RMSE'].mean()
        avg_rmse = np.sqrt(avg_rmse)
        range_summary.append({
            'range': f'{start}-{end}',
            'true_count': true_count,
            'avg_predicted_count': range_df['predicted_count'].mean(),
            'average_MAE': round(avg_mae, 8),
            'average_RMSE': round(avg_rmse, 8),
            'frame_count': len(range_df)
        })

    range_df = pd.DataFrame(range_summary)
    per_range_csv = os.path.join(output_dir, 'rangewise_accuracy_summary.csv')
    range_df.to_csv(per_range_csv, index=False)
    print(f"✅ Saved per-range summary to: {per_range_csv}")

    return frame_df, range_df

def visualize_mae_rmse(frame_df, range_df, output_dir):
    """
    Visualizes MAE/RMSE from per-frame and per-range accuracy metrics.

    Args:
        frame_df (pd.DataFrame): Output of `calculate_framewise_mae_rmse`, per-frame accuracy.
        range_df (pd.DataFrame): Output of `calculate_framewise_mae_rmse`, per-range summary.
        output_dir (str): Directory to save plots.
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- Line Plot: Frame-wise MAE & RMSE ---
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='frame_num', y='MAE', data=frame_df, label='MAE', color='tab:blue')
    sns.lineplot(x='frame_num', y='RMSE', data=frame_df, label='RMSE', color='tab:orange')
    plt.title('Frame-wise MAE and RMSE')
    plt.xlabel('Frame Number')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, 'framewise_mae_rmse.png')
    plt.savefig(plot_path1)
    plt.close()
    print(f"✅ Saved frame-wise error plot to: {plot_path1}")

    # --- Bar Chart: Range-wise MAE & RMSE ---
    melted = range_df.melt(id_vars='range', value_vars=['average_MAE', 'average_RMSE'],
                           var_name='Metric', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='range', y='Value', hue='Metric', palette='Set2')
    plt.title('Average MAE and RMSE per Ground Truth Range')
    plt.xlabel('Frame Range')
    plt.ylabel('Error')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, 'rangewise_mae_rmse.png')
    plt.savefig(plot_path2)
    plt.close()
    print(f"✅ Saved range-wise summary plot to: {plot_path2}")


def run_detection():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = r'received_data\CSV'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = get_unique_filename('detection_results', 'csv', output_dir)

    rack_model = YOLO(args.rack_model, task='detect').to(device)
    obj_model = YOLO(args.obj_model, task='detect').to(device)
    rack_labels = rack_model.names
    obj_labels = obj_model.names

    img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    vid_ext = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

    source_type = determine_source_type(args.source, img_ext, vid_ext)

    resize = False
    if args.resolution:
        resize = True
        resW, resH = map(int, args.resolution.split('x'))

    if args.record:
        recorder, resW, resH = init_recorder(source_type, args.resolution)
    else:
        recorder = None

    imgs_list, cap = prepare_input_source(source_type, args.source, img_ext, args.resolution)

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                  (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    avg_fps = 0
    fps_buffer = []
    fps_len = 100
    img_count = 0
    frame_idx = 0
    csv_data = []
    frame_summary_data = []  #  NEW
    track_class_history = defaultdict(lambda: deque(maxlen=30))

    # NEW: Create folder for saving annotated frames
    frame_output_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frame_output_dir, exist_ok=True)

    while True:
        t_start = time.perf_counter()
        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list): break
            frame = cv2.imread(imgs_list[img_count]); img_count += 1
        else:
            ret, frame = cap.read()
            if not ret: break

        frame_id_str = f"frame_{frame_idx}"
        frame_idx += 1

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        rack_results = rack_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        rack_boxes = rack_results[0].boxes
        rack_data = []

        for box in rack_boxes:
            xyxy = box.xyxy.cpu().numpy().squeeze()
            ymin, ymax, xmin, xmax = int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])
            rack_data.append((ymin, ymax, xmin, xmax))

        rack_data.sort(key=lambda x: x[0])

        for i, (ymin, ymax, xmin, xmax) in enumerate(rack_data):
            label = f"Rack {i+1}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        obj_infer_start = time.perf_counter()
        obj_results = obj_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        obj_infer_time = time.perf_counter() - obj_infer_start
        obj_boxes = obj_results[0].boxes
        obj_data = []

        for box in obj_boxes:
            xyxy = box.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            original_cls_id = int(box.cls.item())
            conf = box.conf.item()
            if conf < args.thresh:
                continue
            track_id = int(box.id.item()) if box.id is not None else -1

            if track_id != -1:
                track_class_history[track_id].append(original_cls_id)
                cls_id = Counter(track_class_history[track_id]).most_common(1)[0][0]
            else:
                cls_id = original_cls_id

            obj_data.append((cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id))

        racks_objs = [[] for _ in range(len(rack_data))]

        for obj in obj_data:
            cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id = obj
            for rack_idx, (rack_ymin, rack_ymax, _, _) in enumerate(rack_data):
                if rack_ymin <= cy <= rack_ymax:
                    racks_objs[rack_idx].append(obj)
                    break

        for rack_objs in racks_objs:
            rack_objs.sort(key=lambda x: x[0], reverse=True)

        obj_count = 0
        for rack_idx, rack_objs in enumerate(racks_objs):
            for i, (cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id) in enumerate(rack_objs):
                color = bbox_colors[cls_id % len(bbox_colors)]
                label = f"{track_id}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                obj_count += 1

                csv_data.append({
                    "frame_id": frame_id_str,
                    "rack_id": f"Rack_{rack_idx+1}",
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": obj_labels[cls_id],
                    "confidence": conf,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "cx": cx,
                    "cy": cy
                })

        # NEW
        class_counter = Counter()
        for obj in obj_data:
            cls_id = obj[6]
            class_name = obj_labels[cls_id]
            class_counter[class_name] += 1

        # NEW
        total_objects_in_frame = sum(class_counter.values())
        for class_name, count in class_counter.items():
            frame_summary_data.append({
                'frame_id': frame_id_str,
                'class_name': class_name,
                'count': count,
                'total_objects': total_objects_in_frame
            })

        fps = 1.0 / (time.perf_counter() - t_start)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Objects: {obj_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ✅ NEW: Save annotated frame
        frame_path = os.path.join(frame_output_dir, f"{frame_id_str}.jpg")
        cv2.imwrite(frame_path, frame)

        cv2.imshow("Rack & Object Detection", frame)
        if recorder: 
            recorder.write(frame)
            print("Recording frame...")

        key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
        if key == ord('q'): break
        elif key == ord('s'): cv2.waitKey()
        elif key == ord('p'): cv2.imwrite('capture.png', frame)

    if cap: cap.release()
    if recorder: recorder.release()
    cv2.destroyAllWindows()

    detection_df = pd.DataFrame(csv_data)

    # print("\nMost Common Classes by Track ID:")
    # for track_id in sorted(track_class_history.keys()):
    #     history = track_class_history[track_id]
    #     most_common = Counter(history).most_common(1)[0]
    #     class_id = most_common[0]
    #     class_name = obj_labels.get(class_id, f"Unknown_{class_id}")
    #     print(f"  Track {track_id}: {class_name} (Class ID: {class_id}) | Count: {most_common[1]}/{len(history)}")

    
    # ✅ Save per-frame class summary
    summary_df = pd.DataFrame(frame_summary_data)
    summary_csv_path = os.path.join(output_dir, "frame_class_counts.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved per-frame class summary to: {summary_csv_path}")

    ground_truth_ranges = [
    (0, 80, 61),
    (81, 99, 62),
    (100, 114, 63),
    (115, 140, 65),
    (141, 203, 66)
    ]   

    framewise_accuracy_df, rangewise_accuracy_df = calculate_framewise_mae_rmse(
        summary_df, ground_truth_ranges, output_dir
    )
    
    # Visualization
    visualize_mae_rmse(framewise_accuracy_df, rangewise_accuracy_df, output_dir)


    print(f"Results saved to: {csv_file}")
    print(f"Average FPS: {avg_fps:.2f}")

    log_summary(
    output_dir=output_dir,
    total_frames=frame_idx,
    total_objects=obj_count,
    total_racks=len(rack_data),
    csv_file=csv_file,
    avg_fps=avg_fps,
    inference_time=obj_infer_time
    )

    return detection_df

# Second part -> post processing
def filter_by_rack_and_group_by_frame(detection_df):
    df_rack1 = pd.DataFrame(columns=detection_df.columns)
    df_rack2 = pd.DataFrame(columns=detection_df.columns)
    df_rack3 = pd.DataFrame(columns=detection_df.columns)

    frame_ids = detection_df['frame_id'].unique()

    for frame_id in frame_ids:
        frame_df = detection_df[detection_df['frame_id'] == frame_id]

        if 'Rack_1' in frame_df['rack_id'].values:
            df_rack1 = pd.concat([df_rack1, frame_df[frame_df['rack_id'] == 'Rack_1']], ignore_index=True)
            

        if 'Rack_2' in frame_df['rack_id'].values:
            df_rack2 = pd.concat([df_rack2, frame_df[frame_df['rack_id'] == 'Rack_2']], ignore_index=True)
            

        if 'Rack_3' in frame_df['rack_id'].values:
            df_rack3 = pd.concat([df_rack3, frame_df[frame_df['rack_id'] == 'Rack_3']], ignore_index=True)
            

    return df_rack1, df_rack2, df_rack3

# Planogram generation

def generate_planogram_df_from_racks(df_rack1, df_rack2, df_rack3):
    """
    Generates a planogram DataFrame from three rack DataFrames by:
    - Concatenating them
    - Keeping only track_ids that appear in ≥30 frames
    - Assigning the most frequent class_name for each track_id
    - Inserting blank rows between rack_id changes and below the header

    Args:
        df_rack1, df_rack2, df_rack3 (pd.DataFrame): DataFrames with detection results.

    Returns:
        pd.DataFrame: Formatted planogram DataFrame.
    """
    from collections import Counter

    # Step 1: Combine all racks
    df_all = pd.concat([df_rack1, df_rack2, df_rack3], ignore_index=True)

    # Step 2: Filter and build valid entries
    valid_entries = []
    valid_detections = [] 
    
    for track_id in df_all["track_id"].unique():
        filtered_rows = df_all[df_all["track_id"] == track_id]
        
        if len(filtered_rows) >=30:
            valid_detections.append(filtered_rows)
            # Most common class_name and class_id for this track
            class_name = Counter(filtered_rows["class_name"]).most_common(1)[0][0]
            class_id = Counter(filtered_rows["class_id"]).most_common(1)[0][0]
            rack_id = filtered_rows.iloc[0]["rack_id"]
            valid_entries.append({
                "track_id": track_id,
                "rack_id": rack_id,
                "class_id": class_id,
                "class_name": class_name
            })
    
    planogram_df = pd.DataFrame(valid_entries)
    
    valid_detections_df = pd.concat(valid_detections, ignore_index=True)
    valid_detections_df.to_csv("valid_planogram_detections.csv", index=False)

    # Step 3: Insert blank rows
    rows_with_blanks = []
    rows_with_blanks.append({col: "" for col in planogram_df.columns})  # blank row after header

    last_rack = None
    for _, row in planogram_df.iterrows():
        current_rack = row["rack_id"]
        if last_rack is not None and current_rack != last_rack:
            rows_with_blanks.append({col: "" for col in planogram_df.columns})
        rows_with_blanks.append(row)
        last_rack = current_rack

    return pd.DataFrame(rows_with_blanks)



def split_row(df,index,empty):
    # Error checking 
    if index < 0 or index >=len(df):
        raise IndexError("Index our of range!")
    
    # Extract row that is empty - not used currently
    row_to_move = df.iloc[index]
    rack_id = row_to_move['rack_id']
    
    # Remove the row
    df_dropped = df.drop(index)

    # Create empty rows
    empty_rows = pd.DataFrame([{col: 'empty' for col in df.columns}] * empty)
    empty_rows['rack_id'] = rack_id
    
    # Index for inseting empty row
    insertion_index = index

    # Split dropped df to insert empty row
    top = df_dropped.iloc[:insertion_index]
    bottom = df_dropped[insertion_index:]
    
    # create new dataframe
    new_df = pd.concat([top, empty_rows, bottom])

    return new_df

def planogram_analysis(generated_df, expected_df):
    generated_df.replace('', np.nan, inplace=True)
    pd.set_option('future.no_silent_downcasting', True)
    
    # Find the empty rows from generated_df
    empty_rows_index = generated_df[generated_df['class_name'] == 'empty'].index
    
    
    # Empty list to store index info
    neighbour_info = []

    # iterate index and keep index of before and after & error checking
    for idx in empty_rows_index:
        if idx - 1 >= 0:
            before_class = generated_df.loc[idx -1, 'class_name']
            rack_id = generated_df.loc[idx, 'rack_id']
        else: 
            None
        if idx + 1 < len(generated_df):
            after_class = generated_df.loc[idx + 1, 'class_name']
            rack_id = generated_df.loc[idx, 'rack_id']
        else:
            None

        neighbour_info.append([idx, before_class, after_class, rack_id])
        

    df_to_change = generated_df
    last_index = 0
    
    for idx, before_class, after_class, rack_id in neighbour_info:
        #! idx is index in generated_df 
        idx += last_index 
        
        matching_indices_before = expected_df[expected_df['class_name'] == before_class].index.tolist() # Find last occurance of before_class -> [-1]
        matching_indices_after = expected_df[expected_df['class_name'] == after_class].index.tolist() # Find first occurance of after_class -> [0]

        matching_indices_before_generated = generated_df[generated_df['class_name'] == before_class].index.tolist() # Find last occurance of before_class in generated -> [-1]
        matching_indices_after_generated = generated_df[generated_df['class_name'] == after_class].index.tolist() # Find first occurance of after_class in generated -> [0]
        
        if pd.isna(before_class) and after_class: # If first item is missing
            # Partially missing algo
            
            next_item_index = matching_indices_after[-1]+1
            
            next_item_index_generated = matching_indices_after_generated[-1]+1
            
            item_count_expected = 0 # number of item counts for first in expected df 
            while pd.notna(expected_df.loc[next_item_index-item_count_expected]['class_name']):
                item_count_expected +=1

            item_count_generated = 0 # number of item counts for first in generated df
            while pd.notna(generated_df.loc[next_item_index_generated-item_count_generated]['class_name']) and not generated_df.loc[next_item_index_generated-item_count_generated]['class_name'] == 'empty':
                item_count_generated +=1
            
            # Compare generated and expected numbers
            # Insert empty row at top of row
            last_index += abs(item_count_expected-item_count_generated) - 1 
            df_to_change = split_row(df_to_change,idx,item_count_expected-item_count_generated) 
            df_to_change = df_to_change.reset_index(drop=True)
            
            
            

        elif pd.isna(after_class) and before_class:
            next_item_index = matching_indices_before[-1]-1

            next_item_index_generated = matching_indices_before_generated[0]-1

            item_count_expected = 0 # number of item counts for first in expected df 
            while pd.notna(expected_df.loc[next_item_index-item_count_expected]['class_name']):
                item_count_expected +=1

            item_count_generated = 0 # number of item counts for first in generated df
            while pd.notna(generated_df.loc[next_item_index_generated-item_count_generated]['class_name']) and not generated_df.loc[next_item_index_generated-item_count_generated]['class_name'] == 'empty':
                item_count_generated +=1

        # elif before_class == after_class:
        # # within class missing
        #     total_index_value = abs(matching_indices_before[-1]-matching_indices_after[0])
        #     total_item_expected = total_index_value + 1
        #     print(f"Quantity : {quantity_of_empty}")
        #     last_index += quantity_of_empty - 1
        #     df_to_change = split_row(df_to_change,idx,quantity_of_empty)
        #     df_to_change = df_to_change.reset_index(drop=True)
        elif before_class == after_class:
            # within class missing
            total_expected_items = len(matching_indices_after)
            total_generated_items = len(matching_indices_after_generated)
            quantity_of_empty = abs(total_expected_items - total_generated_items)
            last_index += quantity_of_empty - 1
            df_to_change = split_row(df_to_change,idx,quantity_of_empty)
            df_to_change = df_to_change.reset_index(drop=True)

        else:
            # Fully missing algo in middle of row
            quantity_of_empty = abs(matching_indices_before[-1]-matching_indices_after[0])-1
            
            if quantity_of_empty == 0:
                quantity_of_empty = 1
            last_index += quantity_of_empty - 1
            df_to_change = split_row(df_to_change,idx,quantity_of_empty)
            df_to_change = df_to_change.reset_index(drop=True)

        #! Fully missing algo needed
        
    df_to_change.to_csv('received_data/CSV/main_case.csv', index=False)
    return df_to_change

def count_missing_expected_products(result_df):
    # Filter rows where output is 'EMPTY' (case-insensitive)
    empty_mask = result_df['output'].str.upper() == 'EMPTY'
    
    # Filter the rows that are empty
    empty_rows = result_df[empty_mask]
    
    # Count the expected_class_name occurrences among empty slots
    missing_counts = empty_rows['expected_class_name'].value_counts().to_dict()
    
    return missing_counts


def check_planogram(generated_df, expected_df, output_excel_path=None):
    gen = generated_df.rename(columns={'class_name': 'generated_class_name'})
    exp = expected_df.rename(columns={'class_name': 'expected_class_name'})

    combined_rows = []
    rack_ids = gen['rack_id'].unique()

    for rack in rack_ids:
        gen_rack = gen[gen['rack_id'] == rack].reset_index(drop=True)
        exp_rack = exp[exp['rack_id'] == rack].reset_index(drop=True)

        max_len = max(len(gen_rack), len(exp_rack))

        # Pad shorter with empty rows
        if len(gen_rack) < max_len:
            empty_rows = pd.DataFrame({
                'rack_id': [rack] * (max_len - len(gen_rack)),
                'generated_class_name': ['empty'] * (max_len - len(gen_rack))
            })
            gen_rack = pd.concat([gen_rack, empty_rows], ignore_index=True)

        if len(exp_rack) < max_len:
            empty_rows = pd.DataFrame({
                'rack_id': [rack] * (max_len - len(exp_rack)),
                'expected_class_name': ['empty'] * (max_len - len(exp_rack))
            })
            exp_rack = pd.concat([exp_rack, empty_rows], ignore_index=True)

        for i in range(max_len):
            gen_cls = gen_rack.at[i, 'generated_class_name']
            exp_cls = exp_rack.at[i, 'expected_class_name']

            if gen_cls == 'empty' or exp_cls == 'empty':
                output = 'EMPTY'
            elif gen_cls != exp_cls:
                output = 'MISPLACED'
            else:
                output = ''

            combined_rows.append({
                'rack_id': rack,
                'generated_class_name': gen_cls,
                'expected_class_name': exp_cls,
                'output': output
            })

        combined_rows.append({'rack_id': '', 'generated_class_name': '', 'expected_class_name': '', 'output': ''})

    # Remove trailing blank row
    if combined_rows and all(value == '' for value in combined_rows[-1].values()):
        combined_rows.pop()

    result_df = pd.DataFrame(combined_rows)

    missing_summary_dict = count_missing_expected_products(result_df)

    print("Missing product counts:", missing_summary_dict)


    if output_excel_path:
        def highlight_output(s):
            colors = []
            for v in s:
                if v == 'RED':
                    colors.append('background-color: red')
                elif v == 'YELLOW':
                    colors.append('background-color: yellow')
                else:
                    colors.append('')
            return colors

        styled = result_df.style.apply(highlight_output, subset=['output'])
        styled.to_excel(output_excel_path, index=False)
        print(f"Saved colored output to {output_excel_path}")

    return result_df, missing_summary_dict

def create_missing_report(missing_summary_dict):
    # Convert the dict to a DataFrame with two columns: missing_class_name, missing_count
    report_df = pd.DataFrame({
        'missing_class_name': list(missing_summary_dict.keys()),
        'missing_count': list(missing_summary_dict.values())
    })
    
    # Optionally, sort by count descending
    report_df = report_df.sort_values(by='missing_count', ascending=False).reset_index(drop=True)
    
    return report_df

def count_detected_classes(df):
    """
    Counts the number of occurrences of each detected class in the detection DataFrame,
    excluding empty strings and the 'empty' placeholder.

    Args:
        df (pd.DataFrame): DataFrame containing detection results with a 'class_name' column.

    Returns:
        pd.DataFrame: A DataFrame showing class_name and count, sorted by count descending.
    """
    # Remove empty strings and 'empty'
    filtered_df = df[df['class_name'].astype(str).str.strip().ne('')]
    filtered_df = filtered_df[filtered_df['class_name'].str.lower() != 'empty']

    # Count occurrences
    class_counts = filtered_df['class_name'].value_counts().reset_index()
    class_counts.columns = ['class_name', 'count']

    # Add total row
    total = class_counts['count'].sum()
    class_counts.loc[len(class_counts)] = ['TOTAL', total]

    print("\n--- Detected Class Counts (excluding empty) ---")
    print(class_counts.to_string(index=False))
    return class_counts




def compare_class_counts(count_df_generated, count_df_expected):
    """
    Compares class counts between generated and expected DataFrames, including TOTAL row,
    and calculates per-class and overall detection accuracy.

    Args:
        count_df_generated (pd.DataFrame): DataFrame with ['class_name', 'count'] from generated data.
        count_df_expected (pd.DataFrame): DataFrame with ['class_name', 'count'] from expected data.

    Returns:
        pd.DataFrame: DataFrame showing class_name, expected_count, generated_count, missing_count, accuracy (%).
    """
    # Merge on class_name
    merged = pd.merge(
        count_df_expected, count_df_generated,
        on='class_name', how='outer', suffixes=('_expected', '_generated')
    )

    # Fill missing values
    merged['count_expected'] = merged['count_expected'].fillna(0).astype(int)
    merged['count_generated'] = merged['count_generated'].fillna(0).astype(int)

    # Compute missing and accuracy
    merged['missing_count'] = merged['count_expected'] - merged['count_generated']
    merged['accuracy (%)'] = merged.apply(
        lambda row: (row['count_generated'] / row['count_expected'] * 100)
        if row['count_expected'] > 0 else 0,
        axis=1
    ).round(2)

    # Calculate overall accuracy using TOTAL row
    total_row = merged[merged['class_name'].str.upper() == 'TOTAL']
    if not total_row.empty:
        total_expected = total_row['count_expected'].values[0]
        total_generated = total_row['count_generated'].values[0]
        overall_accuracy = (total_generated / total_expected) * 100 if total_expected > 0 else 0
        print(f"\n✅ Overall Detection Accuracy: {overall_accuracy:.2f}%")
    else:
        print("\n⚠️ TOTAL row not found. Cannot compute overall accuracy.")

    print("\n--- Missing Class Comparison with Accuracy ---")
    print(merged[merged['missing_count'] > 0].to_string(index=False))

    return merged




if __name__ == '__main__':

    expected_planogram_df = pd.read_excel(r'received_data\Planogram\planogram_misplaced.xlsx')
    detection_df = run_detection()

    df_rack1, df_rack2, df_rack3 = filter_by_rack_and_group_by_frame(detection_df)

    planogram_df = generate_planogram_df_from_racks(df_rack1, df_rack2, df_rack3)

    count_df_generated = count_detected_classes(planogram_df)
    count_df_generated.to_csv("received_data/CSV/class_count_summary_generated.csv", index=False)

    count_df = count_detected_classes(expected_planogram_df)
    count_df.to_csv("received_data/CSV/class_count_summary_expected.csv", index=False)

    new_generated_df = planogram_analysis(planogram_df, expected_planogram_df)
    comparison_df,missing_summary_dict = check_planogram(new_generated_df, expected_planogram_df, "received_data/CSV/planogram_comparison.xlsx")

    missing_report_df = create_missing_report(missing_summary_dict)
    missing_report_df.to_csv("received_data/CSV/missing_report.csv", index=False)

    missing_class_report = compare_class_counts(count_df_generated, count_df)
    missing_class_report.to_csv("received_data/CSV/missing_class_count_comparison.csv", index=False)

    

    
    
