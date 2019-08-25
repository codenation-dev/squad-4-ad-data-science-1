Steps for run:

Prepare environment:

$ pip3 install virtualenv
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt

You need to have in your directory (workspace/data):
- One csv file named ‘portfolio.csv’
- One csv file named ‘market.csv’
- One csv file named ‘dicionário.csv’

Run:
python main_cross_recommender.py 