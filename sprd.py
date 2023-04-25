import pandas as pd
import numpy as np
from mwmatching import maxWeightMatching
from IPython.display import display

class SwissPowerRankDraw():

    def __init__(self, round_count=3):
        self.round_cols = ['team_name_x', 'team_name_y', 'strength_x', 'strength_y', 'margin']
        self.round_count = round_count

    def first_round(self, teams):
        half = teams.shape[0] // 2
        first_half = teams.head(half).reset_index(drop=True).add_suffix('_x')
        second_half = teams.tail(half).reset_index(drop=True).add_suffix('_y')
        first_round =  pd.concat([first_half, second_half], axis=1)
        return first_round


    def get_rankings(self, rounds):

        def get_team_ranks(tourn_df, team1_col, team2_col, margin_col):

            def get_game_df(tourn_df, team1_col, team2_col):
                team1_dummies = pd.get_dummies(pd.concat([tourn_df[team1_col], tourn_df[team2_col]])).head(tourn_df.shape[0])
                team2_dummies = pd.get_dummies(pd.concat([tourn_df[team1_col], tourn_df[team2_col]])).tail(tourn_df.shape[0])
                team1_games = team1_dummies + team2_dummies*-1
                return team1_games

            def get_team_strengths(game_df, margin_col):
                A = game_df.values
                y = tourn_df[margin_col].values
                pinv = np.linalg.pinv(A)
                return pd.Series(pinv.dot(y), index=game_df.columns).sort_values(ascending=False)
            
            def predict_margin(game_df, strengths):
                margin = game_df.dot(strengths)
                margin.name = 'Predicted margin'
                return margin

            game_df = get_game_df(tourn_df, team1_col, team2_col)
            team_strengths = get_team_strengths(game_df, margin_col)
            expected_margin = predict_margin(game_df, team_strengths)

            return team_strengths, expected_margin


        # return df of team and strength and rank
        team_strengths, expected_margin = get_team_ranks(rounds, 'team_name_x', 'team_name_y', 'margin')
        team_strengths = (team_strengths.reset_index()
                          .rename(columns={'index':'team_name', 0:'Strength'}))


        # ranks are used for indicies therefore must start from 0
        team_strengths['rank'] = team_strengths['Strength'].rank(ascending=False, method = 'first').round(0).astype(int)-1
        return team_strengths, expected_margin

    def calculate_next_round(self, rounds):

        team_ranks, _ = self.get_rankings(rounds)

        previous_pairs = rounds.apply(lambda row: frozenset([row["team_name_x"], row["team_name_y"]]), axis=1)
        new_pairs = self.find_pairings(team_ranks, previous_pairs)

        return new_pairs

    def find_pairings(self, team_strengths, previous_pairs):

        # get all combos
        all_combos = self.get_all_combos(team_strengths)

        # create distances
        all_combos = self.create_distances(all_combos, previous_pairs)

        # find best pairings
        new_round = self.get_best_pairings(all_combos, team_strengths)
        return new_round

    def get_all_combos(self, team_strengths):

        team_strengths['key'] = 1
        all_combos = team_strengths.merge(team_strengths, on='key')

        #drop self play
        all_combos = all_combos.loc[all_combos['team_name_x'] != all_combos['team_name_y'], :]

        #drop reciprecals
        all_combos['pairings'] = all_combos.apply(lambda row: frozenset([row['team_name_x'], row['team_name_y']]), axis=1)
        all_combos = all_combos.drop_duplicates(subset = ['pairings'])

        return all_combos

    def create_distances(self, all_combos, previous_pairs):

        all_combos['has-played'] = all_combos['pairings'].isin(previous_pairs)

        # calculate distance, invert could be changed
        close_rank = (all_combos['rank_x'] - all_combos['rank_y']).abs() ** 1.1
        close_rank_max = max(close_rank)
        avoid_replays = np.where(all_combos['has-played'], 0,close_rank_max)
        close_rank_invert = close_rank_max + 1 - close_rank

        # final calculations
        all_combos['distances'] = avoid_replays + close_rank_invert

        return all_combos

    def get_best_pairings(self, all_combos, team_strengths):

        tuples = all_combos.apply(lambda row: (row['rank_x'], row['rank_y'], row['distances']), axis=1)
        pairs = maxWeightMatching(tuples.to_list(),  maxcardinality=False)

        team_strengths = team_strengths.sort_values(['Strength'], ascending=False)
        team_strengths['rank'] = team_strengths.reset_index(drop=True).index
        team_look_up = team_strengths[['team_name']].reset_index(drop=True).to_dict()


        #convert to round format
        new_round_df = pd.Series(pairs, name='team_name_y').reset_index().rename(columns={'index':'team_name_x'})
        new_round_df['team_name_x'] = new_round_df['team_name_x'].map(team_look_up['team_name'])
        new_round_df['team_name_y'] = new_round_df['team_name_y'].map(team_look_up['team_name'])
        new_round_df['pairings'] = new_round_df.apply(lambda row: frozenset([row['team_name_x'], row['team_name_y']]), axis=1)
        new_round_df = new_round_df.drop_duplicates(subset = ['pairings'])
        new_round_df = new_round_df.drop(columns = 'pairings')

        return new_round_df

    def add_team_strength(self, teams, new_round):

        teams_x = teams[['team_name', 'strength']].rename(columns = {'team_name': 'team_name_x', 'strength': 'strength_x'})
        teams_y = teams[['team_name', 'strength']].rename(columns = {'team_name': 'team_name_y', 'strength': 'strength_y'})

        new_round = new_round.merge(teams_x, how='left', on='team_name_x')
        new_round = new_round.merge(teams_y, how='left', on='team_name_y')

        return new_round

    def run_seeding_rounds(self, teams, results_function):
        # teams is sorted in the initial order and contains team strength values
        teams = teams.sort_values('inital_rank')

        # run first round
        rounds = self.first_round(teams)
        rounds['margin'] = rounds.apply(lambda row: results_function(row['strength_x'], row['strength_y']), axis = 1)
        rounds = rounds.loc[:, self.round_cols]

        # run more rounds
        for i in range(self.round_count-1):

            # calculate next round
            new_round = self.calculate_next_round(rounds)

            # calculate next round results
            # add team strength
            new_round = self.add_team_strength(teams, new_round)
            new_round['margin'] = new_round.apply(lambda row: results_function(row['strength_x'], row['strength_y']), axis = 1)

            #update rounds
            rounds = pd.concat([rounds, new_round])

        team_ranks = self.get_rankings(rounds)
        top_8 = team_ranks.sort_values('Strength', ascending=False).head(8)

        return top_8, rounds

def throw_user_error(message):
    display(message)
    input("Press enter to proceed...")
    assert False, message

def RunRound(original_df, check_with_user = True):
    '''Takes in a spreadsheet of results and interactively calculates the next round'''
    
    spread_sheet_cols = original_df.columns


    if not set(['team_name_x', 'team_x_score', 'team_name_y', 'team_y_score']).issubset(spread_sheet_cols):
        throw_user_error('Please make sure all 4 columns are present and start again')

    tournament_so_far = original_df.copy()
    tournament_so_far = tournament_so_far.dropna()

    # check with user
    display(tournament_so_far)
    
    if check_with_user:
        check_read_is_correct = input("Is the above results correct and complete (Y/N):")
        if check_read_is_correct != 'Y': throw_user_error('Please check your spreadsheet and run the code again')

    tournament_so_far['margin'] = tournament_so_far['team_x_score'] - tournament_so_far['team_y_score']

    # calculate team ranks
    sdt = SwissPowerRankDraw()
    team_ranks, expected_margin = sdt.get_rankings(tournament_so_far)
    display('Current Rankings')
    display(team_ranks)

    next_round = sdt.calculate_next_round(tournament_so_far)
    display('Next Round')
    display(next_round)

    display('Spreadsheet Update')
    tournament = pd.concat([original_df, next_round])
    display(tournament)

    if check_with_user:
        check_read_overwrite_ready = input("Are you happy with this update?: (Y/N)")
        if check_read_overwrite_ready != 'Y': throw_user_error('Spreadsheet overwrite cancelled, please run again')
            
    return tournament[spread_sheet_cols], team_ranks, expected_margin

