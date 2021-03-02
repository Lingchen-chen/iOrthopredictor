import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# https://drive.google.com/file/d/1f9XQ0WERs7zdxu_V42pS16cozcUMB4cj/view?usp=sharing
file_id = '1f9XQ0WERs7zdxu_V42pS16cozcUMB4cj'
file_name = 'vgg_19.zip'
chpt_path = './extern/vgg/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)

destination = os.path.join(chpt_path, file_name)
gdd.download_file_from_google_drive(file_id=file_id,
									dest_path=destination,
									unzip=True)
