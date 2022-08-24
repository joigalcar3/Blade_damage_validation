import os
import pandas as pd
import shutil


def change_file_name(folder, file, new_folder):
    file_path = os.path.join(folder, file)
    contents = pd.read_csv(file_path)
    comments = contents.keys()
    alpha_angle = -1
    wind_speed = -1
    blade_damage = -1
    for comment in comments:
        if "alpha" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" d")
            alpha_angle = int(comment[index_equal+1:index_space])
        if "wind_speed" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" m")
            wind_speed = int(comment[index_equal+1:index_space])
        if "blade_damage" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" %")
            blade_damage = int(comment[index_equal+1:index_space])

    if "no propeller" in comments[0]:
        new_file_name = f"a{alpha_angle}_w{wind_speed}.csv"
        new_folder = os.path.join(os.path.dirname(new_folder), "No_propeller")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_file_path = os.path.join(new_folder, new_file_name)
    else:
        if blade_damage == -1:
            blade_damage = 0
        new_file_name = f"b{blade_damage}_a{alpha_angle}_w{wind_speed}.csv"
        new_file_path = os.path.join(new_folder, new_file_name)

    if os.path.exists(new_file_path):
        raise ValueError(f"{new_file_name} already exists")

    shutil.copy2(file_path, new_file_path)


if __name__ == "__main__":
    wd = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign" \
         "\\Data\\Test files"
    nwd = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign" \
          "\\Data\\Test_files_correct_names"
    filenames = os.listdir(wd)
    for f in filenames:
        change_file_name(wd, f, nwd)
