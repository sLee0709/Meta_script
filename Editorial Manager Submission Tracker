import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import html5lib
from getpass import getpass
from tabulate import tabulate

print('Welcome to Submission Check System (v1.0.beta). If you have any questions please contact me through: lishen0709@gmail.com')
time.sleep(2)
print('Please wait for initiation...')
for i in range(5):
    print('.')
    time.sleep(1)

with requests.session() as s:
    print('1. Read your local account information.')
    print('2. Enter manually.')
    print('Please select 1 or 2:')
    usr_selection = int(input(''))
    if usr_selection == 1:
        account_path = './accounts.xlsx'
        account = pd.read_excel(account_path, header=0)
        account_4_show = pd.read_excel(account_path, header=0)
        account_4_show['Password'] = account_4_show['Password'].apply(lambda x: '***')
        account_nums = len(account)
        print('You have ' + str(account_nums) + ' account record(s) in your local file, which one do you want to select?')
        account_table = tabulate(account_4_show, headers=list(account.columns), tablefmt='grid')
        print(account_table)
        account_selection = input("Enter here:")

        usrname = account.iloc[int(account_selection), 0]
        psd = account.iloc[int(account_selection), 1]
        tar_j = account.iloc[int(account_selection), 2]
        tar_j_full = account.iloc[int(account_selection), 3]
        print('Your journal selection is: '+str(tar_j_full))
        time.sleep(2)

    else:
        print('Please input the abbreviation of your targeted journal, For example, you can input "tranon" for Translational Oncology')
        tar_j = str(input('Input here:'))
        print('Please enter your login information:')
        usrname = str(input('Username:'))
        psd = str(getpass('Password:'))

    url = 'https://www.editorialmanager.com/' + tar_j + '/LoginAction.ashx'
    auth = {'username': usrname, 'password': psd}
    print('Connecting...')
    r = s.post(url=url, data=auth)
    mainmenu = "parent.location.href = 'Default.aspx?pg=AuthorMainMenu.aspx'"
    if mainmenu in r.text:
        print('Successfully connected! Please wait for further information...')
        url2 = 'https://www.editorialmanager.com/' + tar_j + '/AuthorMainMenu.aspx'
        r2 = s.get(url2)
        soup = BeautifulSoup(r2.text, 'html.parser')
        count_list = [0]
        status_list = []
        counts = soup.find_all(name='span', attrs={'class': 'count'})
        status = soup.find_all(attrs={'cssclass': 'main_menu_item_2'})
        pattern_count = re.compile(r'[(](.*?)[)]', re.S)
        pattern_status = re.compile(r'[>](.*?)[<]', re.S)
        for num in counts:
            count_list.append(int(re.findall(pattern_count, str(num))[0]))
        for stat in status:
            item = re.findall(pattern_status, str(stat))[0]
            new_item = item.replace('\r\n', '')
            status_list.append(new_item)
        data_dict = {'ManuscriptStatus': status_list, 'Counts': count_list}
        print('Here is an overview of your main menu:')
        time.sleep(2)
        print(pd.DataFrame(data_dict))
        time.sleep(2)
        if sum(count_list) == 0:
            print("It seems you don't have any on-going submissions...")
            input('Press ENTER to exit this session...')
            exit()
        print('You can visit the following sections for more details:')
        active_status = soup.find_all(name='a', attrs={'cssclass': 'main_menu_item_2'})
        if active_status:
            active_status.pop(0)  # 移除submit new manuscript
            active_status_list = []
            active_status_link_list = []
            for item in active_status:
                link = item['href']
                new_link = 'https://www.editorialmanager.com/' + tar_j + '/' + str(link)
                item = re.findall(pattern_status, str(item))[0]
                new_item = item.replace('\r\n', '')
                active_status_list.append(new_item)
                active_status_link_list.append(new_link)
            for c in range(len(active_status_list)):
                print(active_status_list[c] + '\t' + 'No:' + str(c + 1))
            print('please select your action (by no.):')
            usr_selection = int(input('')) - 1
            tar_url = active_status_link_list[usr_selection]
            r3 = s.get(tar_url)
            tar_soup = BeautifulSoup(r3.text, 'html5lib')
            table = tar_soup.find("table", id="datatable", class_="datatable")
            columns = table.find_all("th")
            columns = [c.text.strip() for c in columns]
            rows = []
            for row in table.find("tbody").find_all("tr"):
                cells = row.find_all("td")
                cells = [c.text.strip() for c in cells]
                rows.append(cells)
            tar_df = pd.DataFrame(rows, columns=columns)
            tar_df['Title'] = tar_df['Title'].apply(lambda x: str(x[0:10]) + '...' if len(x) > 10 else x)
            tar_df = tar_df.drop("Action", axis=1)
            # tar_df = tar_df.drop("Title", axis=1)
            final_tar_df = tabulate(tar_df, headers=list(tar_df.columns), tablefmt='grid')
            print(final_tar_df)
        input('Press "enter" to exit.')
    else:
        print('Login failed! Please try it later.')
