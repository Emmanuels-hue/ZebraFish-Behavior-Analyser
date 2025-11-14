Zebrafish Behavior Analyzer
The Zebrafish Behavior Analyzer is a Windows desktop application built in Python for extracting key behavioral metrics from zebrafish swimming videos. It simplifies the process of video-based behavior tracking using motion detection and analysis techniques.

What It Does
Tracks zebrafish movement from clean top-down videos (one fish per video).
Measures key behavioral parameters such as:
Speed and distance traveled
Freezing percentage and movement index
Duration, frame count, and FPS

Supports both single-video analysis and pre–post comparison to evaluate changes before and after an intervention.
Automatically generates motion plots and bar-chart comparisons using Matplotlib.
Exports all results in a CSV file for easy reporting or further analysis.

Features
Simple and intuitive Tkinter GUI for loading one or two videos.
Lightweight and optimized for fast video processing with OpenCV and NumPy.
Available as a standalone Windows executable (.exe) under GitHub Releases.
Full Python source code included for users who prefer manual execution or customization.

Use Cases
Behavioral analysis experiments
Drug-response and neuroactivity studies
Academic or lab-based data collection

Tech Stack
Language: Python
Libraries: OpenCV, NumPy, Matplotlib, Tkinter
Platform: Windows (standalone .exe and source code versions available)

***DISCLAIMER -  The screeshots are from when i performed pre and post comparison analysis using same video (A goldfish swimming with black background : |https://www.youtube.com/watch?v=zZD-7o97ugk&list=PPSV|) with pre, being the video at normal speed and post being the same video at 2X speed. This was because i couldn't find Zebrafish video online , so went on to perform it using goldfish.***

<img width="472" height="249" alt="Screenshot 2025-11-14 120921" src="https://github.com/user-attachments/assets/f9e65abf-2916-47e7-ad14-b56c590186f1" />
<img width="1919" height="684" alt="Screenshot 2025-11-14 120849" src="https://github.com/user-attachments/assets/c54282ef-3a73-4f21-b1a1-135450fcb345" />
<img width="1893" height="257" alt="Screenshot 2025-11-14 120844" src="https://github.com/user-attachments/assets/76d746fa-17b3-4f0b-ac15-2a34f4d98f94" />
<img width="1115" height="585" alt="Screenshot 2025-11-14 120834" src="https://github.com/user-attachments/assets/ea04e511-14e4-4103-947c-7442d4456a9f" />
<img width="870" height="585" alt="Screenshot 2025-11-14 120825" src="https://github.com/user-attachments/assets/1a0e8280-a3c3-45a4-92ff-0527a068c5fa" />
<img width="1913" height="744" alt="Screenshot 2025-11-14 120816" src="https://github.com/user-attachments/assets/5adb3610-9ed3-4d93-aca3-1a7e67c8d066" />
<img width="1919" height="691" alt="Screenshot 2025-11-14 120348" src="https://github.com/user-attachments/assets/67f125ee-2521-43b1-b0e3-3df998b38f22" />
