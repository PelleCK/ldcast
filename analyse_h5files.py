import h5py
import os

# Replace 'your_file.h5' with the path to your .h5 file
file_path = r'D:\Documents\UNI\Master\THESIS weather forecasting\Models\ldcast\data\demo\20210622\RZC211731820VL.801.h5'
cb = ((128,480), (160,608))

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    # List all datasets in the HDF5 file
    R = file["dataset1"]["data1"]["data"][:]
    R = R[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1]]
    print(R.shape)

def read_knmi_data(
        data_dir
        ):
    all_files = os.listdir(data_dir)

    # Filter out only files (excluding directories)
    files_only = [file for file in all_files if os.path.isfile(os.path.join(data_dir, file))]

    # Take the first four files
    first_four_files = files_only[:4]

    # Print the first four files
    for file in first_four_files:
        print(file)
        with h5py.File(os.path.join(data_dir, file), 'r') as file:
            # print("Datasets contained within the file:")
            # for name in file:
            #     print(name)
            
            # Access a dataset and print its shape
            # Replace 'dataset_name' with the name of your dataset
            dataset_name = 'image1'
            if dataset_name in file:
                data = file[dataset_name]['image_data']
                print(type(data))
                print(data.shape)
            else:
                print(f"Dataset '{dataset_name}' not found in the file.")
        break

read_knmi_data(r'D:\Documents\UNI\Master\THESIS weather forecasting\data\dataset-download')
