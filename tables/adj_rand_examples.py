from changeforest_simulations import adjusted_rand_score

for true_changepoints, estimated_changepoints, comment in [
    # ([0, 50, 100, 150], [0, 50, 100, 150]),
    # ([0, 50, 100, 150], [0, 52, 99, 150]),
    # ([0, 50, 100, 150], [0, 23, 50, 100, 150]),
    # ([0, 50, 100, 150], [0, 43, 87, 97, 150]),
    # ([0, 50, 100, 150], [0, 50, 150]),
    # ([0, 50, 100, 150], [0, 23, 150]),
    ([0, 17, 46, 55, 68, 144, 214], [0, 17, 46, 55, 68, 144, 214], "perfect fit"),
    (
        [0, 17, 46, 55, 68, 144, 214],
        [0, 15, 45, 55, 68, 142, 214],
        "almost perfect fit",
    ),
    ([0, 17, 46, 55, 68, 144, 214], [0, 46, 55, 68, 144, 214], "undersegmentation"),
    ([0, 17, 46, 55, 68, 144, 214], [0, 17, 46, 55, 144, 214], "undersegmentation"),
    (
        [0, 17, 46, 55, 68, 144, 214],
        [0, 17, 46, 55, 68, 80, 144, 214],
        "oversegmentation",
    ),
    (
        [0, 17, 46, 55, 68, 144, 214],
        [0, 17, 46, 55, 68, 100, 144, 214],
        "oversegmentation",
    ),
    ([0, 17, 46, 55, 68, 144, 214], [0, 50, 100, 150, 214], "random segmentation"),
    ([0, 17, 46, 55, 68, 144, 214], [0, 214], "no segmentation"),
]:
    print(
        f"{', '.join(map(str, true_changepoints))} & {', '.join(map(str, estimated_changepoints))} & {adjusted_rand_score(true_changepoints, estimated_changepoints):.2f} & {comment} \\\\"
    )
