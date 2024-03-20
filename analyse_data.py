import netCDF4 as nc

# Path to your .nc file
nc_file = r'D:\Documents\UNI\Master\THESIS weather forecasting\LDCast data\RZC\patches_RZC_201805.nc'

# Open the .nc file
ds = nc.Dataset(nc_file, mode='r')

# Print global attributes
print("Global Attributes:")
for name in ds.ncattrs():
    print(f"{name}: {ds.getncattr(name)}")

print("\nVariables:")
# Print variables and their details
for var in ds.variables:
    print(f"\n{var}")
    print(f"Dimensions: {ds.variables[var].dimensions}")
    print(f"Size: {ds.variables[var].size}")
    print(f"Shape: {ds.variables[var].shape}")
    print(f"Data type: {ds.variables[var].dtype}")
    for attr in ds.variables[var].ncattrs():
        print(f"{attr}: {ds.variables[var].getncattr(attr)}")

# Close the dataset
ds.close()
