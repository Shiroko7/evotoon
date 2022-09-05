# evotoon

- evotoon.py is the library module, where each function used for tuning is defined and explained.

- data_classes.py has structures/classes used for representing data.

- main.ipynb has running examples of the evotoon library.

- prototype.ipynb has an example of tunning the ANT colony algorithm for the knapsack problem.

- everything above is broken but evotoon-setup-acotsp is an example of command line usage that works

- comments on setup txt files are denoted by ##

Requirements:
- Python 3.8+
- numpy
- pandas
- matplotlib
- tensorflow
- keras

Instalation:
pip install -r requirements.txt

Run:
python evotoon.py setup_path SEED

IE:
python evotoon.py evotoon-setup-acotsp/ 123

