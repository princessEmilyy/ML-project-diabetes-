#Install environment modules with a loop due to different version of Python

#Read original pip requirements txt
req_txt = pd.read_csv("pip_requirements_yahel.txt",header=None)

#Split the package name from the version and keep only the name
req_txt = req_txt[0].str.split("==",expand = True)[0]

#Save as new txt file withot the version numbers
req_txt.to_csv('pip_requirements.txt',sep = '\t',index = False)

#Convert to DataFrame from np series
req_txt = pd.DataFrame(req_txt)
req_txt.columns = ["package_name"]

#Loop through the DataFrame and install packages
for index, row in req_txt.iterrows():
    package_name = row['package_name']
    try:
        subprocess.check_call(["pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package '{package_name}': {e}")