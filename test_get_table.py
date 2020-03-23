import pict2colony as p2c
import colony2growth as c2g
import growth2ngrowth as g2n
import configparser
import pickle

if __name__ == '__main__':
  config = configparser.ConfigParser()
  config.read('pict2colony.ini')
  p2c.get_colony_table(config)
  # with open('colony_table.pkl', 'rb') as f:
  #   colony_table = pickle.load(f)
  # growth_talbe = c2g.get_growth_table(colony_table)
  # ngrowth_table = g2n.get_ngrowth_table(growth_talbe)
  # print(ngrowth_table)