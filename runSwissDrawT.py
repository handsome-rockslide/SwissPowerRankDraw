import pandas as pd
import numpy as np
from mwmatching import maxWeightMatching
from datetime import datetime
from sprd import *
from IPython.display import display
import openpyxl

if __name__ == '__main__':
    display('Please close TournamentSpreadsheet.xlsx')
    
    spread_sheet_name = 'TournamentSpreadsheet.xlsx'

    original_df = pd.read_excel(spread_sheet_name, sheet_name = 'Rounds')

    results, ranks, expected_margin = RunRound(original_df, True)
    
    analysis_info = pd.concat([original_df, expected_margin], axis= 1)
    
    try:
        with pd.ExcelWriter(spread_sheet_name) as writer:  
            results.to_excel(writer, sheet_name='Rounds', index=False)
            ranks.to_excel(writer, sheet_name='CurrentRanks', index=False)
            analysis_info.to_excel(writer, sheet_name='Analysis_Info', index=False)

        display('TournamentSpreadsheet.xlsx is ready for the results from the next round')
    except PermissionError as e:
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_save = 'TournamentSpreadsheet' + now_str + '.xlsx'
        display('Can\'t access TournamentSpreadsheet.xlsx saving as ' + new_save)
        
        with pd.ExcelWriter(new_save) as writer:  
            results.to_excel(writer, sheet_name='Rounds', index=False)
            ranks.to_excel(writer, sheet_name='CurrentRanks', index=False)
            analysis_info.to_excel(writer, sheet_name='Analysis_Info', index=False)
        
    
    input("Press enter to proceed...")