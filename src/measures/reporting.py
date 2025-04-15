# Prints out summary metrics for that game along with returning the acquired metrics.

import pandas as pd
import numpy as np
from src.measures.get_ids import calc_state_summary, calc_ids, calc_updates, calc_statistics


def game_metrics(path, verbose = True):
	updated_df = pd.read_json(path)

	state_summary = calc_state_summary([x['state'] for x in updated_df['episode']])
	score = updated_df['episode'][len(updated_df['episode'])-1]['score']
	updates = calc_updates(state_summary)
	updates_with_id, item_list = calc_ids(updates)
	sample_stats = calc_statistics(item_list, time_coef=updated_df['total_time'][0] / len(updated_df))

	if verbose:
		print("Percentage Contribution:\t", sample_stats[0])
		print("Average Time Between Actions:\t", np.mean(sample_stats[1]))
		print("Specialization:\t\t\t", np.mean([np.max(sample_stats[2][0]) /
										  (np.sum(sample_stats[2][0]) + 1),
										  np.max(sample_stats[2][1]) /
										  (np.sum(sample_stats[2][1]) + 1)]))
		print('Total Movement Actions Taken:\t', np.sum(sample_stats[3]))
		print('Wasted Actions:\t\t\t', len(sample_stats[4]))
		print('Final Score:\t\t\t', score)
	return tuple(list(calc_statistics(item_list)) + [score] + [updated_df['total_time'][0]])