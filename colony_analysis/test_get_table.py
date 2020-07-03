import colony_analysis.pict2colony as p2c
import colony_analysis.colony2growth as c2g
import colony_analysis.growth2ngrowth as g2n
import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('pict2colony.ini')
    colony_table = p2c.get_colony_table(config)
    growth_talbe = c2g.get_growth_table(colony_table)
    ngrowth_table = g2n.get_ngrowth_table(growth_talbe)
