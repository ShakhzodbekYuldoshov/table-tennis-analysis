config = {
    "table_points": {
        "top_left": (217*2, 239*2),
        "top_right": (751*2, 258*2),
        "bottom_left": (79*2, 403*2),
        "bottom_right": (878*2, 433*2),
        "left_side_points": {
            "top_left": (217*2, 239*2),
            "bottom_left": (79*2, 403*2),
            "top_right": (486*2, 248*2),
            "bottom_right": (483*2, 420*2)
        },
        "right_side_points": {
            "top_left": (486*2, 248*2),
            "bottom_left": (483*2, 420*2),
            "top_right": (751*2, 258*2),
            "bottom_right": (879*2, 433*2)
        }
    },
    "net_area": {
        # Net is in the center between left and right table sides
        # These coordinates define a rectangular area where the net is located
        "top_left": (470*2, 200*2),      # Above the table center
        "top_right": (500*2, 205*2),     # Above the table center
        "bottom_left": (475*2-30, 440*2),   # Below the table center
        "bottom_right": (495*2+30, 445*2)   # Below the table center
    }
}