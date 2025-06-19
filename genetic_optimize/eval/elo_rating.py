import random
import math
from typing import List

from genetic_optimize.TFparamsBase import TFparamsBase

class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.rating = 1000  # 初始ELO积分
        self.matches = set()  # 记录对战过的对手ID

    def __repr__(self):
        return f"Player {self.id} (ELO: {self.rating:.1f}, Skill: {self.true_skill:.1f})"

def elo_update(winner: TFparamsBase, loser: TFparamsBase, k=32):
    # ELO积分计算公式
    expected = 1 / (1 + 10 ** ((loser.rating - winner.rating) / 400))
    delta = k * (1 - expected)
    winner.rating += delta
    loser.rating -= delta

def swiss_pairing(players: List[TFparamsBase]):
    # 按积分降序排列
    sorted_players = sorted(players, key=lambda x: -x.rating)
    paired = set()
    pairs = []

    for i in range(len(sorted_players)):
        if i in paired: continue
        
        # 寻找最近未交战的对手
        for j in range(i+1, len(sorted_players)):
            if j in paired: continue
            if sorted_players[j].id not in sorted_players[i].matches:
                pairs.append((sorted_players[i], sorted_players[j]))
                paired.update([i, j])
                break
        else:  # 没有可用对手时随机匹配
            for j in range(i+1, len(sorted_players)):
                if j not in paired:
                    pairs.append((sorted_players[i], sorted_players[j]))
                    paired.update([i, j])
                    break
    return pairs

def simulate_match(p1, p2):
    # 根据真实实力决定胜负（实际场景中不可见）
    if p1.true_skill > p2.true_skill:
        winner, loser = p1, p2
    elif p1.true_skill < p2.true_skill:
        winner, loser = p2, p1
    else:  # 平局处理
        winner, loser = random.choice([(p1, p2), (p2, p1)])
    
    # 更新ELO积分
    elo_update(winner, loser)
    
    # 记录对战历史
    p1.matches.add(p2.id)
    p2.matches.add(p1.id)

def tournament(n=8):
    # 初始化选手
    players = [Player(i) for i in range(n)]
    
    # 计算所需轮次（log2(n) + 2轮）
    rounds = int(math.log2(n)) + 2
    
    # 初始随机配对
    random.shuffle(players)
    for i in range(0, n, 2):
        p1, p2 = players[i], players[i+1]
        simulate_match(p1, p2)

    # 瑞士轮阶段
    for _ in range(rounds-1):
        pairs = swiss_pairing(players)
        for p1, p2 in pairs:
            simulate_match(p1, p2)

    # 最终排名
    final_ranking = sorted(players, key=lambda x: (-x.rating, -x.true_skill))
    return final_ranking

# 运行示例
if __name__ == "__main__":
    final = tournament(8)
    print("最终排名（ELO积分 + 真实实力）：")
    for i, p in enumerate(final, 1):
        print(f"{i:2}. {p}")