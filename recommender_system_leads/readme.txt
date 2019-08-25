Steps for run:

# Prepare environment

For first use:
$ pip3 install virtualenv
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
$ mkdir -p workspace/data/

You need to have these files in your directory (workspace/data):
- ‘estaticos_portfolio1.csv’
- ‘estaticos_portfolio2.csv’
- ‘estaticos_portfolio3.csv’
- ‘estaticos_market.csv’
- ‘dicionário.csv’

Link to download above files:
https://drive.google.com/open?id=18Wb6BLSu7ls6S5z_uu8ob0h-x6n2N14b


# To Run, change code with portfolio info:

In "main_ranking.py":
- Lines 10 to 15, select ranking files to compare and aggregate;

Run:
$ python main_ranking.py 