import numpy as np
import random


def majority_votes(votes):
    """
    count votes and return winner
    """
    count_vote = {}
    
    for x in votes:
        count_vote[x] = count_vote.get(x, 0) + 1
    
 
    winners = []
    max_count = max(count_vote.values())
    
    for vote,count in count_vote.items():
        if count == max_count:
            winners.append(vote)
        
    return random.choice(winners)

import scipy.stats as ss


def majority_votes_short(votes):
    """
    count votes and return winner
    """
    mode, count = ss.mode(votes)
    return mode


def distance(pe1,p2):
    return np.sqrt(np.sum(np.power(p2-p1, 2)))

p1 = np.array([1,1])
p2 = np.array([4,4])

ans = distance(p1, p2)
print(ans)

votes = [1,2,3,4,5,5,4,3,2,2,4,4,6,3,3]
# winner = majority_votes(votes)
winner = majority_votes_short(votes)
print(*winner)
