import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# https://drive.google.com/file/d/10rgyg4R6Yn1_uyGzLU0WRJBresrdqFCG/view?usp=sharing
file_id = '10rgyg4R6Yn1_uyGzLU0WRJBresrdqFCG'
file_name = 'examples.zip'
chpt_path = './'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)

destination = os.path.join(chpt_path, file_name)
gdd.download_file_from_google_drive(file_id=file_id,
									dest_path=destination,
									unzip=True)
