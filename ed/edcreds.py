def getedcreds():

    # change your credentials here, do not push them to github! 
    uid = '<your_nasa_earthdata_user_id>'
    pwd = '<your_nasa_earthdata_password>'
    email = '<your_nasa_earthdata_account_email>'

    # to print a message if they haven't been changed
    if uid == '<your_nasa_earthdata_user_id>':
        print('\n WARNING: YOU NEED TO SET UP YOUR NASA EARTHDATA CREDENTIALS TO DOWNLOAD ICESAT-2 DATA!\n')
        print('  update the info in ed/edcreds.py :\n')
        print("  def getedcreds():")
        print("    # change your credentials here, do not push them to github!")
        print("    uid = '<your_nasa_earthdata_user_id>'")
        print("    pwd = '<your_nasa_earthdata_password>'")
        print("    email = '<your_nasa_earthdata_account_email>'")
        return None
    else:
        return uid, pwd, email
    