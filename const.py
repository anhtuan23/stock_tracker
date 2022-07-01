ACC1_NAME = "acc1"
ACC2_NAME = "acc2"

TN_USER_NAME = "tn"

ACC_USER_DICT = {
    ACC1_NAME: ["ta", TN_USER_NAME, "at"],
    ACC2_NAME: ["tl", "la"],
}
USER_NAME_FEE_DICT = {
    TN_USER_NAME: 0.1,
}
USER_NAME_FEE_ANCHOR_DATE = {
    TN_USER_NAME: "2022-07-01",
}

ACC_NAME_L = list(ACC_USER_DICT.keys())
ACC_COMBINED_NAME = "acc_combined"
ALL_ACC_NAME_L = ACC_NAME_L + [ACC_COMBINED_NAME]

VNI_NAME = "vni"
VN30_NAME = "vn30"
INDEX_NAME_L = [VNI_NAME, VN30_NAME]
INDEX_COMBINED_NAME = "index_combined"
ALL_INDEX_NAME_L = INDEX_NAME_L + [INDEX_COMBINED_NAME]

COLOR_LIST = [
    "dodgerblue",
    "darkorange",
    "darkgreen",
    "darkviolet",
    "aqua",
    "tomato",
    "peru",
    "mediumspringgreen",
    "magenta",
]
