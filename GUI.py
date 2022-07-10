from train_and_validate import *

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

def create_df_from_pickle(pickle_file_path, pc):
    with open(pickle_file_path, "rb") as file:
        metrics = pickle.load(file)
    df = pd.DataFrame(columns=[i for i in range(1, 6)], index=[pc])
    df.loc[pc] = metrics
    df["Mean"] = [np.mean(metrics)]
    df["Median"] = [np.median(metrics)]
    return df

def display_df(pickle_file_path_1,
               pickle_file_path_2,
               pickle_file_path_3,
               pickle_file_path_4,
               text_widget):
    pickle_file_paths = (pickle_file_path_1.get(), pickle_file_path_2.get(), pickle_file_path_3.get(), pickle_file_path_4.get())
    pc_sequence = ["Original", "CLAHE", "UM", "HEF"]
    dataframes = []
    for i, path in enumerate(pickle_file_paths):
        if path != "":
            pc = pc_sequence[i]
            df = create_df_from_pickle(path, pc)
            dataframes.append(df)
        else:
            continue
    if len(dataframes) > 1:
        metrics_df = pd.concat(dataframes)
    else:
        metrics_df = dataframes[0]
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, str(metrics_df))

def train_and_validate(variables, pc1_value,
                       pc2_value,
                       pc3_value,
                       pc4_value,
                       model_opt1_combobox,
                       model_opt2_combobox,
                       hparam1_entry,
                       hparam2_entry,
                       hparam3_entry,
                       hparam4_entry,
                       hparam5_entry,
                       hparam6_entry,
                       hparam7_entry,
                       weight_path,
                       metrics_path):

    pc_state = (pc1_value.get(), pc2_value.get(), pc3_value.get(), pc4_value.get())
    base_model = model_opt1_combobox.get()
    transfer_learning = model_opt2_combobox.get()
    batch_size = int(hparam1_entry.get())
    epochs = int(hparam2_entry.get())
    loss_function = hparam3_entry.get()
    optimizer = hparam4_entry.get()
    learning_rate = float(hparam5_entry.get())
    class_num = int(hparam6_entry.get())
    if hparam7_entry.get() == "224x224":
        input_size = 224
    weight_path = weight_path.get()
    metrics_path = metrics_path.get()
    for i, value in enumerate(pc_state):
        if value == 0:
            continue
        if i == 0:
            dataset = "original"
        elif i == 1:
            dataset = "CLAHE"
        elif i == 2:
            dataset = "UM"
        if i == 3:
            dataset = "HEF"
        train_eval_function(base_model,
                            dataset,
                            batch_size,
                            epochs,
                            loss_function,
                            optimizer,
                            learning_rate,
                            class_num,
                            input_size,
                            transfer_learning,
                            variables,
                            weight_path,
                            metrics_path)

def create_askdir_dialogue(title, variable):
    path = filedialog.askdirectory(initialdir="/", title=title)
    variable.set(path)

def create_combobox(container, values, state):
    combobox = ttk.Combobox(container)
    combobox["values"] = values
    combobox["state"] = state
    return combobox

def display_chosen_pc(combobox_var, algorithm, label):
    if combobox_var.get() == 1:
        label.config(text=f"Latih model dengan dataset {algorithm}", fg="green")
    else:
        label.config(text="")

def display_chosen_metrics_file(combobox_var, metrics_path, metrics_widget, model_widget, algorithm, label, metrics_file_var):
    metrics = metrics_widget.get()
    model = model_widget.get()
    metrics_file = ""
    if combobox_var.get() == 1:
        if model == "MobileNetV2":
            if metrics.lower() == "accuracy":
                metrics_file = os.path.join(metrics_path.get(), f"{algorithm.lower()}_dataset", "mobilenetv2",
                                            f"ex1_{algorithm.lower()}_accuracy.pkl")
            elif metrics.lower() == "f1-score":
                metrics_file = os.path.join(metrics_path.get(), f"{algorithm.lower()}_dataset", "mobilenetv2",
                                            f"ex1_{algorithm.lower()}_f1_score.pkl")

        elif model == "EfficientNet B0":
            if metrics.lower() == "accuracy":
                metrics_file = os.path.join(metrics_path.get(), f"{algorithm.lower()}_dataset", "efficientnet_b0",
                                            f"ex2_{algorithm.lower()}_accuracy.pkl")
            elif metrics.lower() == "f1-score":
                metrics_file = os.path.join(metrics_path.get(), f"{algorithm.lower()}_dataset", "efficientnet_b0",
                                            f"ex2_{algorithm.lower()}_f1_score.pkl")
        if os.path.exists(metrics_file) == False:
            tk.messagebox.showerror(title="Error", message=f"File tidak ditemukan. Anda belum melatih model {model} dengan dataset {algorithm}")
        else:
            label.config(text=f"file dipilih: {metrics_file}", fg="green")
            metrics_file_var.set(metrics_file)
    else:
        label.config(text="")
        metrics_file_var.set("")

def display_messagebox(var):
        if var.get() == "":
            tk.messagebox.showerror(title="Error", message="Anda belum mengisi path ke file metrik")

def create_checkbox(container, text_label, checkbox_var, command):
    checkbox = ttk.Checkbutton(container,
                               text=text_label,
                               command=command,
                               variable=checkbox_var,
                               onvalue=1,
                               offvalue=0)
    return checkbox

def set_grid(num_row, num_column):
    for i in range(0, 30):
        Grid.rowconfigure(main, i, weight=0)
        Grid.columnconfigure(main, i, weight=0)
    for i in range(0, num_row):
        Grid.rowconfigure(main, i, weight=1)
    for i in range(0, num_column):
        Grid.columnconfigure(main, i, weight=1)

def clean_page():
    for i in main.winfo_children():
        i.destroy()

def welcome_page():
    clean_page()
    set_grid(num_row=6, num_column=1)
    img = Image.open("Logo Undip Universitas Diponegoro.png")
    resized_img = img.resize((200, 200), Image.ANTIALIAS)
    resized_img = ImageTk.PhotoImage(resized_img)
    img_label = Label(main, image=resized_img)
    img_label.image = resized_img
    img_label.grid(row=3, column=0, sticky="NSEW", rowspan=1)

    label_1 = Label(main, text="Studi Komparatif Pengaruh Peningkatan Citra Pada",
                    font=("helvetica", 14))
    label_1.grid(row=0, column=0, sticky="NSEW", rowspan=1)
    label_2 = Label(main, text="Klasifikasi Citra Sinar-X Penderita Tuberkulosis",
                    font=("helvetica", 14))
    label_2.grid(row=1, column=0, sticky="NSEW", rowspan=1)
    label_3 = Label(main, text="Menggunakan Jaringan Saraf Konvolusional", font=("helvetica", 14))
    label_3.grid(row=2, column=0, sticky="NSEW", rowspan=1)

    label_NAMA = Label(main, text="Nama: Zainul Muttaqin", font=("helvetica", 14))
    label_NAMA.grid(row=4, column=0, sticky="NSEW", rowspan=1)
    label_NIM = Label(main, text="NIM: 21120118130073", font=("helvetica", 14))
    label_NIM.grid(row=5, column=0, sticky="NSEW", rowspan=1)

    button_NEXT = Button(main, text="Next", command=main_menu_page)
    button_NEXT.config(width=10, height=3)
    button_NEXT.grid(row=6, column=0, sticky="E", padx=20, pady=20, ipadx=5, ipady=5)

def main_menu_page():
    clean_page()
    set_grid(num_row=6, num_column=1)
    title = Label(main, text="Main Menu", font=("helvetica", 14))
    title.grid(row=0, column=0, sticky="NSEW")

    menu_1 = Button(main, text="Training", command=train_page)
    menu_1.config(width=30, height=3)
    menu_1.grid(row=1, column=0, sticky="N", rowspan=1)
    menu_2 = Button(main, text="Training Results", command=training_results_page)
    menu_2.config(width=30, height=3)
    menu_2.grid(row=2, column=0, sticky="N", rowspan=1)
    menu_3 = Button(main, text="Predict", command=None)
    menu_3.config(width=30, height=3)
    menu_3.grid(row=3, column=0, sticky="N", rowspan=1)

    button_BACK = Button(main, text="Back", command=welcome_page)
    button_BACK.config(width=10, height=3)
    button_BACK.grid(row=6, column=0, sticky="W", padx=20, pady=20, ipadx=5, ipady=5)

def train_page():
    clean_page()
    set_grid(num_row=8, num_column=9)
    title = Label(main, text="Training", font=("helvetica", 14))
    title.grid(row=0, column=0, sticky="NSEW", columnspan=9)

    subtitle1 = LabelFrame(main, text="Hyperparameter", font=("helvetica", 11))
    subtitle1.grid(row=1, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    hparam1_label = Label(subtitle1, text="Batch Size") #entry
    hparam1_label.grid(row=0, column=0, sticky="W", pady=2)
    hparam2_label = Label(subtitle1, text="Epochs") #entry
    hparam2_label.grid(row=1, column=0, sticky="W", pady=2)
    hparam3_label = Label(subtitle1, text="Loss Function") #dropdown
    hparam3_label.grid(row=2, column=0, sticky="W", pady=2)
    hparam4_label = Label(subtitle1, text="Optimizer") #dropdown
    hparam4_label.grid(row=3, column=0, sticky="W", pady=2)
    hparam5_label = Label(subtitle1, text="Learning Rate") #entry
    hparam5_label.grid(row=4, column=0, sticky="W", pady=2)
    hparam6_label = Label(subtitle1, text="Jumlah Kelas")  #entry
    hparam6_label.grid(row=5, column=0, sticky="W", pady=2)
    hparam7_label = Label(subtitle1, text="Input Size") #dropdown
    hparam7_label.grid(row=6, column=0, sticky="W", pady=2)

    hparam1_entry = ttk.Entry(subtitle1)
    hparam1_entry.grid(row=0, column=1, sticky="W", pady=2)
    hparam2_entry = ttk.Entry(subtitle1)
    hparam2_entry.grid(row=1, column=1, sticky="W", pady=2)
    hparam3_entry = create_combobox(subtitle1, values=("Categorical Cross Entropy", "Binary Cross Entropy"), state="readonly")
    hparam3_entry.grid(row=2, column=1, sticky="W", pady=2)
    hparam4_entry = create_combobox(subtitle1, values=("Adam", "SGD"), state="readonly")
    hparam4_entry.grid(row=3, column=1, sticky="W", pady=2)
    hparam5_entry = ttk.Entry(subtitle1)
    hparam5_entry.grid(row=4, column=1, sticky="W", pady=2)
    hparam6_entry = ttk.Entry(subtitle1)
    hparam6_entry.grid(row=5, column=1, sticky="W", pady=2)
    hparam7_entry = create_combobox(subtitle1, values=("224x224"), state="readonly")
    hparam7_entry.grid(row=6, column=1, sticky="W", pady=2)

    subtitle2 = LabelFrame(main, text="Model CNN yang digunakan", font=("helvetica", 11))
    subtitle2.grid(row=2, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    model_opt1_label = Label(subtitle2, text="Model")  # dropdown
    model_opt1_label.grid(row=0, column=0, sticky="W", pady=2)
    model_opt2_label = Label(subtitle2, text="Transfer Learning")  # dropdown
    model_opt2_label.grid(row=1, column=0, sticky="W", pady=2)

    model_opt1_combobox = create_combobox(subtitle2, values=("MobileNetV2", "EfficientNet B0"), state="readonly")
    model_opt1_combobox.grid(row=0, column=1, sticky="W", pady=2)
    model_opt2_combobox = create_combobox(subtitle2, values=("Ya", "Tidak"), state="readonly")
    model_opt2_combobox.grid(row=1, column=1, sticky="W", pady=2)

    pc1_value = IntVar()
    pc2_value = IntVar()
    pc3_value = IntVar()
    pc4_value = IntVar()

    subtitle3 = LabelFrame(main, text="Metode peningkatan citra", font=("helvetica", 11))
    subtitle3.grid(row=3, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    pc1_checkbox = create_checkbox(subtitle3, text_label="ORIGINAL", checkbox_var=pc1_value, command=lambda: display_chosen_pc(pc1_value, "Original", pc1_state))
    pc1_checkbox.grid(row=0, column=0, sticky="W", pady=2)
    pc2_checkbox = create_checkbox(subtitle3, text_label="CLAHE", checkbox_var=pc2_value, command=lambda: display_chosen_pc(pc2_value, "CLAHE", pc2_state))
    pc2_checkbox.grid(row=1, column=0, sticky="W", pady=2)
    pc3_checkbox = create_checkbox(subtitle3, text_label="UM", checkbox_var=pc3_value, command=lambda: display_chosen_pc(pc3_value, "UM", pc3_state))
    pc3_checkbox.grid(row=2, column=0, sticky="W", pady=2)
    pc4_checkbox = create_checkbox(subtitle3, text_label="HEF", checkbox_var=pc4_value, command=lambda: display_chosen_pc(pc4_value, "HEF", pc4_state))
    pc4_checkbox.grid(row=3, column=0, sticky="W", pady=2)

    pc1_state = Label(subtitle3, text="")  # dropdown
    pc1_state.grid(row=0, column=1, sticky="W", pady=2)
    pc2_state = Label(subtitle3, text="")  # dropdown
    pc2_state.grid(row=1, column=1, sticky="W", pady=2)
    pc3_state = Label(subtitle3, text="")  # dropdown
    pc3_state.grid(row=2, column=1, sticky="W", pady=2)
    pc4_state = Label(subtitle3, text="")
    pc4_state.grid(row=3, column=1, sticky="W", pady=2)

    subtitle4 = LabelFrame(main, text="Penyimpanan", font=("helvetica", 11))
    subtitle4.grid(row=4, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    save1_label = Label(subtitle4, text="Lokasi penyimpanan bobot")  # entry (path)
    save1_label.grid(row=0, column=0, sticky="W", pady=2)
    save2_label = Label(subtitle4, text="Lokasi penyimpanan metrik")  # entry (path)
    save2_label.grid(row=1, column=0, sticky="W", pady=2)

    weight_path = StringVar()
    save1_entry = ttk.Entry(subtitle4, textvariable=weight_path)
    save1_entry.grid(row=0, column=1, sticky="W", pady=2)
    metrics_path = StringVar()
    save2_entry = ttk.Entry(subtitle4, textvariable=metrics_path)
    save2_entry.grid(row=1, column=1, sticky="W", pady=2)

    save1_button = Button(subtitle4, text="Browse", command=lambda : create_askdir_dialogue(title="Saved weight path", variable=weight_path))
    save1_button.grid(row=0, column=2, sticky="W", ipadx=5, ipady=5)
    save2_button = Button(subtitle4, text="Browse",
                          command=lambda: create_askdir_dialogue(title="Saved metrics path", variable=metrics_path))
    save2_button.grid(row=1, column=2, sticky="W", ipadx=5, ipady=5)

    fold_status = StringVar()
    metrics_status = StringVar()

    button_TRAIN = Button(main,
                          text="Latih",
                          command=lambda: train_and_validate((fold_status, metrics_status), pc1_value, pc2_value, pc3_value, pc4_value,
                                                             model_opt1_combobox, model_opt2_combobox, hparam1_entry, hparam2_entry, hparam3_entry,
                                                             hparam4_entry, hparam5_entry, hparam6_entry, hparam7_entry, weight_path, metrics_path),
                          bg="green")
    button_TRAIN.config(width=5, height=1)
    button_TRAIN.grid(row=5, column=0, sticky="W", padx=20, pady=20, ipadx=5, ipady=5)

    fold_status_label = Label(main, textvariable=fold_status)  # entry (path)
    fold_status_label.grid(row=6, column=0, sticky="W", pady=2, columnspan=9)
    metrics_status_label = Label(main, textvariable=metrics_status)  # entry (path)
    metrics_status_label.grid(row=7, column=0, sticky="W", pady=2, columnspan=9)

    button_BACK = Button(main, text="Back", command=main_menu_page)
    button_BACK.config(width=10, height=3)
    button_BACK.grid(row=5, column=8, sticky="E", padx=20, pady=20, ipadx=5, ipady=5)

def training_results_page():
    clean_page()
    set_grid(num_row=8, num_column=9)
    title = Label(main, text="Training Results", font=("helvetica", 14))
    title.grid(row=0, column=0, sticky="NSEW", columnspan=9)

    subtitle1 = LabelFrame(main, text="Model yang digunakan", font=("helvetica", 11))
    subtitle1.grid(row=1, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    model_opt_label = Label(subtitle1, text="Model: ")  # dropdown
    model_opt_label.grid(row=0, column=0, sticky="W", pady=2)

    model_opt_combobox = create_combobox(subtitle1, values=("MobileNetV2", "EfficientNet B0"),
                                    state="readonly")
    model_opt_combobox.grid(row=0, column=1, sticky="W", pady=2)

    subtitle2 = LabelFrame(main, text="Metrik yang ditampilkan", font=("helvetica", 11))
    subtitle2.grid(row=2, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)
    save1_label = Label(subtitle2, text="Lokasi penyimpanan metrik: ")  # entry (path)
    save1_label.grid(row=0, column=0, sticky="W", pady=2)
    metrics_opt_label = Label(subtitle2, text="Metrik: ")  # dropdown
    metrics_opt_label.grid(row=1, column=0, sticky="W", pady=2)
    metrics_opt_combobox = create_combobox(subtitle2, values=("Accuracy", "F1-Score"),
                                      state="readonly")
    metrics_opt_combobox.grid(row=1, column=1, sticky="W", pady=2)

    metrics_path = StringVar()
    save1_entry = ttk.Entry(subtitle2, textvariable=metrics_path)
    save1_entry.grid(row=0, column=1, sticky="W", pady=2)

    save1_button = Button(subtitle2, text="Browse",
                          command=lambda: create_askdir_dialogue(title="Saved metrics path", variable=metrics_path))
    save1_button.grid(row=0, column=2, sticky="W", ipadx=5, ipady=5, padx=5)

    subtitle3 = LabelFrame(main, text="Metode peningkatan citra", font=("helvetica", 11))
    subtitle3.grid(row=3, column=0, sticky="NSEW", columnspan=9, padx=5, pady=5, ipadx=5, ipady=5)

    pc1_value = IntVar()
    pc2_value = IntVar()
    pc3_value = IntVar()
    pc4_value = IntVar()
    metrics_file_path_1 = StringVar()
    metrics_file_path_2 = StringVar()
    metrics_file_path_3 = StringVar()
    metrics_file_path_4 = StringVar()
    pc1_checkbox = create_checkbox(subtitle3, text_label="ORIGINAL", checkbox_var=pc1_value,
                                   command=lambda: display_chosen_metrics_file(pc1_value, metrics_path, metrics_opt_combobox,
                                                                               model_opt_combobox, "Original", pc1_state,
                                                                               metrics_file_path_1))
    pc1_checkbox.grid(row=0, column=0, sticky="W", pady=2)
    pc2_checkbox = create_checkbox(subtitle3, text_label="CLAHE", checkbox_var=pc2_value,
                                   command=lambda: display_chosen_metrics_file(pc2_value, metrics_path, metrics_opt_combobox,
                                                                               model_opt_combobox, "CLAHE", pc2_state,
                                                                               metrics_file_path_2))
    pc2_checkbox.grid(row=1, column=0, sticky="W", pady=2)
    pc3_checkbox = create_checkbox(subtitle3, text_label="UM", checkbox_var=pc3_value,
                                   command=lambda: display_chosen_metrics_file(pc3_value, metrics_path, metrics_opt_combobox,
                                                                               model_opt_combobox, "UM", pc3_state,
                                                                               metrics_file_path_3))
    pc3_checkbox.grid(row=2, column=0, sticky="W", pady=2)
    pc4_checkbox = create_checkbox(subtitle3, text_label="HEF", checkbox_var=pc4_value,
                                   command=lambda: display_chosen_metrics_file(pc4_value, metrics_path, metrics_opt_combobox,
                                                                               model_opt_combobox, "HEF", pc4_state,
                                                                               metrics_file_path_4))
    pc4_checkbox.grid(row=3, column=0, sticky="W", pady=2)

    pc1_state = Label(subtitle3, text="")  # dropdown
    pc1_state.grid(row=0, column=1, sticky="W", pady=2)
    pc2_state = Label(subtitle3, text="")  # dropdown
    pc2_state.grid(row=1, column=1, sticky="W", pady=2)
    pc3_state = Label(subtitle3, text="")  # dropdown
    pc3_state.grid(row=2, column=1, sticky="W", pady=2)
    pc4_state = Label(subtitle3, text="")
    pc4_state.grid(row=3, column=1, sticky="W", pady=2)

    textbox = Text(main, height=8)
    textbox.grid(row=5, column=0, sticky="NSEW", padx=10, pady=5, columnspan=9)

    button_DISPLAY = Button(main,
                          text="Tampilkan",
                          command=lambda: display_df(metrics_file_path_1, metrics_file_path_2,
                                             metrics_file_path_3, metrics_file_path_4,
                                             textbox),
                          bg="green")
    button_DISPLAY.config(width=10, height=1)
    button_DISPLAY.grid(row=4, column=0, sticky="W", padx=10, pady=5, ipadx=5, ipady=5)

    button_BACK = Button(main, text="Back", command=main_menu_page)
    button_BACK.config(width=10, height=3)
    button_BACK.grid(row=6, column=0, sticky="W", padx=10, pady=10, ipadx=5, ipady=5)

def predict_page():
    clean_page()
    title = Label(main, text="Main Menu", font=("helvetica", 14))
    title.grid(row=0, column=0, sticky="NSEW")

    menu_1 = Button(main, text="Training", command=None)
    menu_1.config(width=30, height=3)
    menu_1.grid(row=1, column=0, sticky="N", rowspan=1)
    menu_2 = Button(main, text="Evaluate", command=None)
    menu_2.config(width=30, height=3)
    menu_2.grid(row=2, column=0, sticky="N", rowspan=1)
    menu_3 = Button(main, text="Predict", command=None)
    menu_3.config(width=30, height=3)
    menu_3.grid(row=3, column=0, sticky="N", rowspan=1)

    button_BACK = Button(main, text="Back", command=welcome_page)
    button_BACK.config(width=10, height=3)
    button_BACK.grid(row=6, column=0, sticky="W", padx=10, pady=20)

main = Tk()
main.geometry("800x700")

welcome_page()

main.mainloop()

