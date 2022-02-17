from changeforest_simulations import adjusted_rand_score

for true_changepoints, estimated_changepoints in [
    ([0, 50, 100, 150], [0, 50, 100, 150]),
    ([0, 50, 100, 150], [0, 52, 99, 150]),
    ([0, 50, 100, 150], [0, 23, 50, 100, 150]),
    ([0, 50, 100, 150], [0, 43, 87, 97, 150]),
    ([0, 50, 100, 150], [0, 50, 150]),
    ([0, 50, 100, 150], [0, 23, 150]),
    # ([0, 17, 46, 55, 68, 144, 214], [0, 17, 55, 144, 214]),
    # ([0, 17, 46, 55, 68, 144, 214], [0, 17, 47, 55, 144, 214]),
    # ([0, 17, 46, 55, 68, 144, 214], [0, 15, 45, 56, 71, 143, 214]),
    # ([0, 17, 46, 55, 68, 144, 214], [0, 55, 214]),
    # ([0, 17, 46, 55, 68, 144, 214], [0, 50, 100, 150, 214]),
]:
    print(
        f"{', '.join(map(str, true_changepoints))} & {', '.join(map(str, estimated_changepoints))} & {adjusted_rand_score(true_changepoints, estimated_changepoints):.2f} \\\\"
    )
