# import zipfile
# path_to_zip_file = "alldata/images_001.tar"
# directory_to_extract_to = "extract"
#
# with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
#     zip_ref.extractall(directory_to_extract_to)

import shutil
import os

# absolute path
src_path = r"alldata/images_002.tar/images_002/images/"
dst_path = r"alldata/test"
# shutil.copy(src_path, dst_path)

def copy(src_path, dst_path):
    if os.path.isdir(dst_path):
        dst = os.path.join(dst_path, os.path.basename(src_path))
    shutil.copyfile(src_path, dst_path)

copy(src_path,dst_path)