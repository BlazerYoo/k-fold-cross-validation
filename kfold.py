k = 10
crs_vld_begin = 0
crs_vld_end = 90
test_begin = 90
test_end = 100


for fold in range(k):
    print(f"FOLD {fold + 1}", end=": ")

    val_end = crs_vld_end - fold*(100/k-1)
    val_start = val_end - 100/k + 1

    if val_end == crs_vld_end:
        train_start = crs_vld_begin
        train_end = val_start
        print(f"Training {train_start}% to {train_end}%")
    elif val_start == crs_vld_begin:
        train_start = val_end
        train_end = crs_vld_end
        print(f"Training {train_start}% to {train_end}%")
    else:
        train_start = crs_vld_begin
        train_mid1 = val_start
        train_mid2 = val_end
        train_end = crs_vld_end
        print(f"Training {train_start}% to {train_mid1}% "
                + f" and {train_mid2} to {train_end}%")


    print(f"\tValidating {val_start}% to {val_end}%\n")

print(f"\nTesting samples from {test_begin}% to {test_end}%")
