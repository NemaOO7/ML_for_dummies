from dataclasses import dataclass
import os
import shutil
import major0_1 as maj1
import major2 as maj2
import major3 as maj3
import cyrene_reverese_predicter as crp
import dataprocessing as clean
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.messagebox import showerror
import pandas as pd
from pandastable import Table
from tkPDFViewer import tkPDFViewer as pdf
from PIL import ImageTk, Image
import sys
import numpy as np
import visualize as g_gen



class PrintLogger(): # create file like object
    def __init__(self, textbox): # pass reference to text widget
        self.textbox = textbox # keep ref

    def write(self, text):
        self.textbox.insert(tk.END, text) # write text to textbox
            # could also scroll to end of textbox here to make sure always visible


class Data:
    def __init__(self, root):
        self.root = root
        self.root.title("Automaton")
        self.root.geometry("1250x630")
        self.root.config(bg="black")
        # self.root.resizable(False, False)
        self.selected_indices = []
        self.frame_step = 0
        self.data = pd.DataFrame()
        self.filename = ''
        self.selected_langs =[]
        self.y_variable = ""
        self.data_object = None
        self.tune = tk.StringVar()
        self.algo_list = []
        self.x_dataframe = None
        self.y_dataframe = None
        self.score = tk.StringVar()
        self.rcflag = tk.StringVar()
        self.utime_mode = tk.StringVar()
        self.cv = tk.StringVar()
        self.seed = tk.StringVar()
        self.iterr = tk.StringVar()
        self.n_jobs = tk.StringVar()
        self.trained_model = ""
        self.main_dict = {}
        self.main_function()
        

    def main_function(self):

        # -----------------------------------------------Welcome frame
        self.welcomeframe = tk.Frame(root, bg='white')
        self.welcomeframe.place(x=0, y=0, width=1250, height=570)
        self.uploadframe = tk.Frame(root, bg='white')
        self.uploadframe.place(x=0, y=0, width=1250, height=570)
        self.encodingframe = tk.Frame(root, bg='white')
        self.encodingframe.place(x=0, y=0, width=1250, height=570)
        self.cleaningframe = tk.Frame(root, bg='white')
        self.cleaningframe.place(x=0, y=0, width=1250, height=570)
        self.algoframe = tk.Frame(root, bg='white')
        self.algoframe.place(x=0, y=0, width=1250, height=570)
        self.reportframe = tk.Frame(root, bg='white')
        self.reportframe.place(x=0, y=0, width=1250, height=570)
        self.control_function()

    def welcome_function(self):
        img = ImageTk.PhotoImage(Image.open("Automaton.jpg"))
        temp = tk.Label(self.welcomeframe, image=img)
        temp.image = img
        temp.pack(padx=10,  expand=tk.YES, fill="both")

    def upload_function(self):
        # -----------------------------------------------Upload frame

        tk.Label(self.uploadframe, text="Upload Dataset File", font=("arial", 40, 'bold'), bg='white').place(x=410, y=150)
        tk.Label(self.uploadframe, text="(example : .csv, .xlsx, .xls)", font=("arial", 12), bg="white").place(x=550, y=220)
        self.file = tk.Label(self.uploadframe, text="No File Choosen", font=("arial", 10), bg="white")
        self.file.place(x=600, y=450)
        tk.Button(self.uploadframe, text="+\nUpload File", font=("arial", 12, 'bold'), bg='#003171', fg='white', command=self.upload_file).place(x=550, y=350, width=200, height=50)

    def continue_clean(self):
        self.data_object = clean.dataprocessing(self.filename)
        self.data_object.xconfig(self.selected_langs, self.y_variable, self.filename)
        self.data_object.summary()
        # if(self.data_object.is_NA(self.filename)):
        #     print("TURE/False ", self.data_object.is_NA())
        #     self.data_object.alertNa(self.filename)
        self.data_object.categoricalcols(self.filename)
        self.data_object.removeoutliner(self.filename)
        self.data = pd.read_csv(self.filename)
        self.data_object.summary()
        self.cleaning_function()
        #data_object.masterfunc(self.filename, self.y_variable.get() ,self.selected_langs)
        #print("FROM TEMP2 - ",data_object.dataset.describe())
        # data_object = clean.dataprocessing(self.filename)
        # data_object.xconfig(self.selected_indices)
        # data_object.main(self.filename, self.y_variable, self.selected_indices)
        # clean.onehotencoding(self.y_variable);
        self.next_frame();

    def encoding_function(self):
        # ------------------------------------------------Encoding frame
        print(len(self.data))
        print(self.data.dropna())
        print(len(self.data))

        # DataFrame
        table_frame = tk.Frame(self.encodingframe)
        table_frame.place(x=0, y=0, width=800, height=570)
        sheet = Table(table_frame, dataframe=self.data)
        sheet.show()
        sheet.place(x=50, y=30, width=800, height=570)

        # tk.Label(self, bg="light blue", text="Feature selection", font=("arial", 20)).place(x=950, y=0)
        list_Frame = tk.LabelFrame(self.encodingframe, text='Select X variables', bg="white", font=('arial', 10, 'bold'))
        list_Frame.place(x=850, y=50, width=350, height=350)
        yscrollbar = tk.Scrollbar(list_Frame)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        columns_var = tk.StringVar(value=list(self.data.columns))
        self.listbox = tk.Listbox(list_Frame, selectmode = "multiple", yscrollcommand = yscrollbar.set, listvariable=columns_var)
        self.listbox.pack(padx=10, pady=10, expand=tk.YES, fill="both")
        self.listbox.bind('<<ListboxSelect>>', self.items_selected)

        #tk.Button(self.encodingframe, text='Continue', bg='#003171', fg='white', font=('arial', 12, 'bold'), command=self.continue_clean()).place(x=850, y=520, width=350)
        # tk.Label(self, bg="white", text="<-- See the preview", font=("arial", 10))
        y_var = tk.StringVar()

        y_var.set("Select Y variable")
        column_list = list(self.data.columns)
        drop= tk.OptionMenu(self.encodingframe, y_var, *column_list, command=self.callback)
        drop.place(x=850, y=440, width=350)
        tk.Button(self.encodingframe, text='Continue', bg='#003171', fg='white', font=('arial', 12, 'bold'), command=self.continue_clean).place(x=850, y=520, width=350)
        
    
    def callback(self, selection):
        self.y_variable = selection
        print("SELF.Y_VARIBEL IS ",self.y_variable)
        #creating y_dataframe
        temp_clean = clean.dataprocessing(self.filename)
        self.y_dataframe = temp_clean.get_y(self.y_variable, self.filename)
        print("TYPE OF Y_VARIABLE IS ",type(self.y_variable))
        print("LENGTH OF Y_VARIABLE IS ", len(self.y_variable))
        print("this is AWWWWSOME ",self.data[[self.y_variable]])

    def tempo(self):
        print("YEH GADBAD HAI")
    def cleaning_function(self):
       # ----------------------------------------------------- Cleaning frame

        # DataFrame
        table_frame = tk.Frame(self.cleaningframe)
        table_frame.place(x=0, y=0, width=800, height=640)
        sheet = Table(table_frame, dataframe=self.data)
        sheet.show()
        sheet.place(x=50, y=30, width=800, height=640)

        list_Frame = tk.LabelFrame(self.cleaningframe, text='List of Null Values', bg="white", font=('arial', 10, 'bold'))
        list_Frame.place(x=850, y=10, width=350, height=160)

        sheet1 = Table(list_Frame, dataframe=pd.read_csv('NULL.csv'))
        sheet1.show()
        sheet1.place(x=50, y=30, width=320, height=160)

        tk.Button(self.cleaningframe, text='Drop', bg='#003171', fg='white', font=('arial', 10, 'bold'), command=self.drop).place(x=870, y=190, width=100)
        tk.Button(self.cleaningframe, text='Mean/Mode', bg='#003171', fg='white', font=('arial', 10, 'bold'), command=self.mean_mode).place(x=1050, y=190, width=100)

        insight_Frame = tk.LabelFrame(self.cleaningframe, text='Data Insights', bg="white", font=('arial', 10, 'bold'))
        insight_Frame.place(x=850, y=250, width=350, height=300)

        sheet = Table(insight_Frame, dataframe=pd.read_csv('insight.csv'))
        sheet.show()
        sheet.place(x=50, y=30, width=800, height=640)

    def drop(self):
        self.data_object.cleaningNA(self.filename)
        self.data_object.cleaningNAChoice(self.filename)
        self.data_object.labelencoding(self.y_variable, self.filename)
        self.data_object.onehotencoding(self.y_variable, self.filename)
        #self.data_object.onehotencoding(self.y_variable)
        self.data = pd.read_csv(self.filename)
        self.cleaning_function()
        print("drop works fine")
        #function(drop the rows) ---->  update dataset file
        #reload cleaning_function

    def mean_mode(self):
        self.data_object.filling_missingdata(self.filename)
        self.data_object.filling_missingdatastring(self.filename)
        self.data_object.labelencoding(self.y_variable, self.filename)
        self.data_object.onehotencoding(self.y_variable, self.filename)
        self.data = pd.read_csv(self.filename)
        self.cleaning_function()
        print("mean_mode works fine")
        #function(mean_mode the rows) ---->  update dataset file
        #reload cleaning_function

    def algo_function(self):
        # -----------------------------------------------------Algo frame
        
        separator = ttk.Separator(self.algoframe, orient='horizontal')
        separator.place(relx=0, rely=0.47, relwidth=1, relheight=1)


        supervised_frame = tk.LabelFrame(self.algoframe, text="Supervised", font=('arial', 12, 'bold'))
        supervised_frame.place(x=40, y=320, width=900, height=220)
        tk.Label(supervised_frame, text="Regression", font=('arial', 12, 'bold')).place(x=30, y=10)        
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('LinearRegression'), text="LinearRegression", font=('arial', 12)).place(x=30, y=40)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('Ridge'), text="Ridge", font=('arial', 12)).place(x=30, y=70)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('Lasso'), text="Lasso", font=('arial', 12)).place(x=30, y=100)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('DecisionTreeRegressor'), text="DecisionTreeRegressor", font=('arial', 12)).place(x=30, y=130)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('SVR'), text="SVR", font=('arial', 12)).place(x=30, y=160)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('KNeighborsRegressor'), text="KNeighborsRegressor", font=('arial', 12)).place(x=240, y=40)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('RandomForestRegressor'), text="RandomForestRegressor", font=('arial', 12)).place(x=240, y=70)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('GradientBoostingRegressor'), text="GradientBoostingRegressor", font=('arial', 12)).place(x=240, y=100)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('AdaBoostRegressor'), text="AdaBoostRegressor", font=('arial', 12)).place(x=240, y=130)

        tk.Label(supervised_frame, text="Classification", font=('arial', 12, 'bold')).place(x=480, y=10)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('KNeighborsClassifier'), text="KNeighborsClassifier", font=('arial', 12)).place(x=480, y=40)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('RandomForestClassifier'), text="RandomForestClassifier", font=('arial', 12)).place(x=480, y=70)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('GradientBoostingClassifier'), text="GradientBoostingClassifier", font=('arial', 12)).place(x=480, y=100)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('AdaBoostClassifier'), text="AdaBoostClassifier", font=('arial', 12)).place(x=480, y=130)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('DecisionTreeClassifier'), text="DecisionTreeClassifier", font=('arial', 12)).place(x=480, y=160)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('SVC'), text="SVC", font=('arial', 12)).place(x=700, y=40)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('LogisticRegression'), text="LogisticRegression", font=('arial', 12)).place(x=700, y=70)
        tk.Checkbutton(supervised_frame, command=lambda:self.create_algo_list('RidgeClassifier'), text="RidgeClassifier", font=('arial', 12)).place(x=700, y=100)



        unsupervised_frame = tk.LabelFrame(self.algoframe, text="Unsupervised", font=('arial', 12, 'bold'))
        unsupervised_frame.place(x=40, y=320, width=900, height=220)
        tk.Checkbutton(unsupervised_frame, command=lambda:self.create_algo_list('Kmean'),text="Kmean", font=('arial', 12)).place(x=30, y=30)
        tk.Button(unsupervised_frame, text='Map', bg='#003171', fg='white', font=('arial', 12), command=self.open_map_window).place(x=170, y=60, width=100)
        tk.Button(unsupervised_frame, text='Input', bg='#003171', fg='white', font=('arial', 12), command=self.input_value_window).place(x=170, y=30, width=100)

        tk.Checkbutton(unsupervised_frame, command=lambda:self.create_algo_list('AgglomerativeClustering'), text="AgglomerativeClustering", font=('arial', 12)).place(x=450, y=30)
        tk.Button(unsupervised_frame, text='Map', bg='#003171', fg='white', font=('arial', 12), command=self.open_map_window).place(x=670, y=60, width=100)
        tk.Button(unsupervised_frame, text='Input', bg='#003171', fg='white', font=('arial', 12), command=self.input_value_window).place(x=670, y=30, width=100)


        inferential_frame = tk.LabelFrame(self.algoframe, text="Inferential", font=('arial', 12, 'bold'))
        inferential_frame.place(x=40, y=320, width=900, height=220)
        tk.Checkbutton(inferential_frame, command=lambda:self.create_algo_list('LinearRegression'), text="LinearRegression", font=('arial', 12)).place(x=30, y=30)
        tk.Button(inferential_frame, text='Input', bg='#003171', fg='white', font=('arial', 12), command=self.input_value_window).place(x=60, y=60, width=100)
        tk.Checkbutton(inferential_frame, command=lambda:self.create_algo_list('LogisticRegression'), text="LogisticRegression", font=('arial', 12)).place(x=450, y=30)
        tk.Button(inferential_frame, text='Input', bg='#003171', fg='white', font=('arial', 12), command=self.input_value_window).place(x=480, y=60, width=100)



        denension_reduchon_frame = tk.LabelFrame(self.algoframe, text="Dimensionality reduction", font=('arial', 12, 'bold'))
        denension_reduchon_frame.place(x=40, y=320, width=900, height=220)
        tk.Checkbutton(denension_reduchon_frame, command=lambda:self.create_algo_list('PCA'), text="PCA", font=('arial', 12)).place(x=30, y=30)
        tk.Checkbutton(denension_reduchon_frame, command=lambda:self.create_algo_list('LinearDiscriminantAnalysis'), text="LinearDiscriminantAnalysis", font=('arial', 12)).place(x=30, y=60)
        tk.Checkbutton(denension_reduchon_frame, command=lambda:self.create_algo_list('kernalPCA'), text="kernalPCA", font=('arial', 12)).place(x=450, y=30)
        
        default_frame = tk.Frame(self.algoframe)
        default_frame.place(x=40, y=320, width=900, height=220)
        tk.Label(default_frame, text="Choose any above algorithms to start learning", font=('arial', 22)).pack(expand=True, fill='both')

        tk.Button(self.algoframe, text='Supervised', bg='#003171', fg='white', font=('arial', 14, 'bold'), command=supervised_frame.tkraise).place(x=40, y=40, width=250, height=180)
        tk.Button(self.algoframe, text='Unsupervised', bg='#003171', fg='white', font=('arial', 14, 'bold'), command=unsupervised_frame.tkraise).place(x=345, y=40, width=250, height=180)
        tk.Button(self.algoframe, text='Inferential', bg='#003171', fg='white', font=('arial', 14, 'bold'), command=inferential_frame.tkraise).place(x=650, y=40, width=250, height=180)
        tk.Button(self.algoframe, text='Dimensionality reduction', bg='#003171', fg='white', font=('arial', 14, 'bold'), command=denension_reduchon_frame.tkraise).place(x=955, y=40, width=250, height=180)
        

        tk.Button(self.algoframe, text='Learn', bg='#003171', fg='white', font=('arial', 12), command=self.open_construtor_window).place(x=1000, y=410, width=210)
        tk.Button(self.algoframe, text='Console', bg='#003171', fg='white', font=('arial', 12), command=self.open_console_window).place(x=1000, y=470, width=210)

    def input_value_window(self):
        top = tk.Toplevel(self.root, width=300, height=150) 
        tk.Label(top, text='Value to be taken :').place(x=0, y=0)
        tk.Entry(top, font=('arial', 17), textvariable=self.tune).place(x=0, y=50, width=300)
        tk.Button(top, text='Save', bg='#003171', fg='white', font=('arial', 12), command=top.destroy).place(x=50, y=100, width=100)

    def report_function(self):
        # ------------------------------------------------------ Report frame
        
        v1 = pdf.ShowPdf()
        v2 = v1.pdf_view(self.reportframe, pdf_location = r"plot1.pdf", bar=True, width=5100,)
        v2.place(x=0, y=0, width=1100, height=640)

        ttk.Button(self.reportframe, text="Download").place(x=1150, y=200)
        ttk.Button(self.reportframe, text="Exit", command=lambda :self.root.quit()).place(x=1150, y=300)

    def start_L(self):
        print(self.tune.get())


    def control_function(self):
        # ======================================================= raise
        self.frame = -1
        self.controlframe = tk.Frame(root, bg='#F3F5F9')
        self.controlframe.place(x=0, y=570, width=1250, height=60)
        tk.Button(self.controlframe, text="Previous", bg='#003171', fg='white', font=('arial', 12, 'bold'), command=self.previous_frame).place(x=500, y=15, width=100)
        if self.frame == 5:
            tk.Button(self.controlframe, text="Exit", bg='#003171', fg='red', font=('arial', 12, 'bold'), command=self.next_frame).place(x=700, y=15, width=100)
        else:
            tk.Button(self.controlframe, text="Next", bg='#003171', fg='white', font=('arial', 12, 'bold'), command=self.next_frame).place(x=700, y=15, width=100)


        
        self.frames = {}
        self.frames[0] = self.welcomeframe
        self.frames[1] = self.uploadframe
        self.frames[2] = self.encodingframe
        self.frames[3] = self.cleaningframe
        self.frames[4] = self.algoframe
        self.frames[5] = self.reportframe

        self.frames_function = ['self.welcome_function()', 'self.upload_function()', 'self.encoding_function()', 'self.cleaning_function()', 'self.algo_function()', 'self.report_function()']

        self.next_frame()

    def next_frame(self):
        self.frame += 1
        frame = self.frames[self.frame]
        frame.tkraise()
        eval(self.frames_function[self.frame])

    
    def previous_frame(self):
        self.frame -= 1
        frame = self.frames[self.frame]
        frame.tkraise()

    def open_map_window(self):
        self.graph()
        top = tk.Toplevel(self.root, width=1250, height=700) 
        tempFrame = tk.Frame(top, bg='lavender')
        tempFrame.place(x=20, y=15, width=1200, height=550)
        img = Image.open("myImagePDF.jpg")
        img = img.resize((1210, 550), Image.ANTIALIAS)
        my_img = ImageTk.PhotoImage(img) 
        temp = tk.Label(tempFrame, image=my_img)
        temp.image = my_img
        temp.pack(padx=10,  expand=tk.YES, fill="both")
    

    def graph(self):
        maj2_obj = maj2.USLearningCore(self.x_dataframe)
        if(self.algo_list[0]=="Kmean"):
            maj2_obj.kmeanslearning()
        else:
            maj2_obj.agglomerativeClustering()
        
    def suggestion_window(self, mess):
        top = tk.Toplevel(self.root, width=50, height=50) 
        tk.Label(top, text=mess).place(x=20, y=10)
        tk.Button(top, text='OK', bg='#003171', fg='white', font=('arial', 12), command=top.destroy).place(x=0, y=30, width=50)

    def open_construtor_window(self):
        self.open_console_window
        if(len(self.algo_list)!=1 and ("SVR" not in self.algo_list or "LinearRegression" not in self.algo_list or "Ridge" not in self.algo_list or "Lasso" not in self.algo_list or "DecisionTreeRegressor" not in self.algo_list or "KNeighbrorsRegressor" not in self.algo_list or "RandomForestRegressor" not in self.algo_list or "GradientBoostingRegressor" not in self.algo_list or "AdaBoostRegressor" not in self.algo_list)):
            mess_age = crp.model_predictor(self.x_dataframe, self.y_dataframe)
            print("The message passsed is : ",mess_age)

        #logic to create the dictionaries of dictionary according to the list of algorithms
            #if(length of list_algos == 1)
                #then fetch the value of list and call the constructor according to it
            #else
                #iterate over the list and append algo_dict(values) to main_dic(algos)

        if(len(self.algo_list)==1):
            algo = self.algo_list[0]
            print("The current algo is : ", algo)
            if(algo=="Kmean"):
                #Kmean constructor
                maj2_obj = maj2.USLearningCore(self.x_dataframe,int(self.tune.get()))
                self.trained_model = maj2_obj.kmeanslearning()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="AgglomerativeClustering"):
                #AgglomerativeClustering constructor
                maj2_obj = maj2.USLearningCore(self.x_dataframe,int(self.tune.get()))
                self.trained_model= maj2_obj.agglomerativeClustering()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="LinearRegression"):
                #LinearRegression constructor
                maj1_obj = maj1.InferentialCore(self.x_dataframe,self.y_dataframe,0,0,bool(self.tune.get()))
                self.trained_model = maj1_obj.startRegression()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="LogisticRegression"):
                #LogisticRegression constructor
                maj1_obj = maj1.InferentialCore(self.x_dataframe,self.y_dataframe,0,0,bool(self.tune.get()))
                self.trained_model = maj1_obj.startLogistic()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="PCA"):
                #PCA constructor
                maj3_obj = maj3.DReductionCore(self.x_dataframe,2)
                self.trained_model = maj3_obj.pcaReducation()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="kernalPCA"):
                #kernalPCA constructor
                maj3_obj = maj3.DReductionCore(self.x_dataframe,2)
                self.trained_model = maj3_obj.kernalpcaReducation()
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            elif(algo=="LinearDiscriminantAnalysis"):
                #LinearDiscriminantAnalysis constructor
                maj3_obj = maj3.DReductionCore(self.x_dataframe,2)
                self.trained_model = maj3_obj.ldaseparation(self.y_dataframe)
                print("THE FINAL MODEL IS ", self.trained_model)
                op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
                op.all_encoder()
                return
            else:
                print("no algo selected")
        else:
            type= ""
            if("SVR" in self.algo_list or "LinearRegression" in self.algo_list or "Ridge" in self.algo_list or "Lasso" in self.algo_list or "DecisionTreeRegressor" in self.algo_list or "KNeighbrorsRegressor" in self.algo_list or "RandomForestRegressor" in self.algo_list or "GradientBoostingRegressor" in self.algo_list or "AdaBoostRegressor" in self.algo_list):
                reg_dict = {}
                type = "Regression"
                for i in self.algo_list:
                    print("The current algo is : ",i)
                    if(i=="LinearRegression"):
                        LR_dict = {'LinearRegression' : {'fit_intercept' : [True,False],'normalize' :[True,False],'copy_X' : [True,False]}}
                        reg_dict.update(LR_dict)
                        #append LinearRegression dictionary to main dictionary
                    elif(i=="Ridge"):
                        #append Ridge dictionary to main dictionary
                        R_dict = {'Ridge' : { 'alpha': [0.01,0.1,1,0.5],'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],'tol':[0.01,0.06,0.1,0.6,1.2,1.6]}}
                        reg_dict.update(R_dict)
                    elif(i=="Lasso"):
                        #append Lasso dictionary to main dictionary
                        La_dict = {'Lasso' :  {'alpha' : [0.01,0.1,1,5,10,0.5],'tol' : [0.01,0.06,0.1,0.6,1.2,1.6],'precompute' : [True , False]}}
                        reg_dict.update(La_dict)
                    elif(i=="DecisionTreeRegressor"):
                        #append DecisionTreeRegressor dictionary to main dictionary
                        DTR_dict = {'DecisionTreeRegressor' :{'min_samples_leaf' :[1,2]}}
                        reg_dict.update(DTR_dict)
                    elif(i=="SVR"):
                        #append SVR dictionary to main dictionary
                        SVR_dict = {'SVR' : {'C' :  [0.01,0.06,0.1,0.6,1.2,1.6],'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],'epsilon' : [0.01,0.06,0.1,0.6,1.2,1.6]}}
                        reg_dict.update(SVR_dict)
                    elif(i=="KNeighbrorsRegressor"):
                        #append KNeighbrorsRegressor dictionary to main dictionary
                        KNR_dict = {'KNeighborsRegressor' : { 'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'weights': ['uniform','distance'],'p':[1,2,5]}}
                        reg_dict.update(KNR_dict)
                    elif(i=="RandomForestRegressor"):
                        #append RandomForestRegressor dictionary to main dictionary
                        RFR_dict =  {'RandomForestRegressor' : {'bootstrap': [True, False],'max_depth': [10, 30, 40, 90, 100],'min_samples_leaf': [1, 2, 4],'min_samples_split': [2, 5, 10],'n_estimators': [200,  600, 800,  1600, 2000]}}
                        reg_dict.update(RFR_dict)
                    elif(i=="GradientBoostingRegressor"):
                        #append GradientBoostingRegressor dictionary to main dictionary
                        GBR_dict = {'GradientBoostingRegressor' :{ 'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],"min_samples_split": np.linspace(0.1, 0.5, 12),"min_samples_leaf": np.linspace(0.1, 0.5, 12),"max_depth":[3,8,15],"max_features":["log2","sqrt"],"criterion": ["friedman_mse",  "mae"],"subsample":[0.5, 0.618, 0.8, 0.9, 0.95, 1.0],"n_estimators":[10,20,60,100]}}
                        reg_dict.update(GBR_dict)
                    elif(i=="AdaBoostRegressor"):
                        #append AdaBoostRegressor dictionary to main dictionary
                        AB_dict = {'AdaBoostRegressor' :{'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],'n_estimators':[10,20,60,100]}}
                        reg_dict.update(AB_dict)
                    else:
                        print("no algo selected")
            else:
                type = "Classification"
                for i in self.algo_list:
                    print("The current algo is : ", i)
                    reg_dict = {}
                    if(i=="LogisticRegression"):
                        #append LogisticRegression dictionary to main dictionary
                        LR_dict = {'LogisticRegression' : { 'penalty': [ 'l1', 'l2', 'elasticnet',None],'C' : [0.01,0.06,0.1,0.6,1.2,1.6]}}
                        reg_dict.update(LR_dict)
                    elif(i=="RidgeClassifier"):
                        #append RidgeClassifier dictionary to main dictionary
                        RC_dict = {'RidgeClassifie' : {'alpha' : [0.01,0.1,1,5,10,0.5],'tol':[0.01,0.06,0.1,0.6,1.2,1.6]}}
                        reg_dict.update(RC_dict)
                    elif(i=="DecisionTreeClassifier"):
                        #append DecisionTreeRegressor dictionary to main dictionary
                        DTR_dict = {'DecisionTreeClassifier' :{'criterion':['gini','entropy']}}
                        reg_dict.update(DTR_dict)
                    elif(i=="SVC"):
                        #append SVC dictionary to main dictionary
                        SVC_dict = {'SVC' :{ 'C':  [0.01,0.06,0.1,0.6,1.2,1.6],'kernel' :['linear', 'poly', 'rbf', 'sigmoid'],'decision_function_shape' : ['ovo', 'ovr']}}
                        reg_dict.update(SVC_dict)
                    elif(i=="KNeighbrorsClassifier"):
                        #append KNeighbrorsRegressor dictionary to main dictionary
                        KNC_dict = {'KNeighborsClassifier' :{ 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20, 21, 22,23, 24, 25, 26, 27, 28, 29, 30],'p':[1,2,5]}}
                        reg_dict.update(KNC_dict)
                    elif(i=="RandomForestClassifier"):
                        #append RandomForestRegressor dictionary to main dictionary
                        RFR_dict = {'RandomForestClassifier' : {'criterion' :["gini", "entropy"],'bootstrap': [True, False],'max_depth': [10, 30, 40, 90, 100, None],'max_features': ['auto','sqrt'],'min_samples_leaf': [1, 2, 4],'min_samples_split': [2, 5, 10],'n_estimators': [200,  600, 800,  1600, 2000]}}
                        reg_dict.update(RFR_dict)
                    elif(i=="GradientBoostingClassifier"):
                        #append GradientBoostingRegressor dictionary to main dictionary
                        GBR_dict = {'GradientBoostingClassifier' : { 'learning_rate' : [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], 'min_samples_split': np.linspace(0.1, 0.5, 12),"min_samples_leaf": np.linspace(0.1, 0.5, 12),"max_depth":[3,8,15],"max_features":["log2","sqrt"],"criterion": ["friedman_mse",  "mae"],'subsample':[0.5, 0.618, 0.8, 0.9, 0.95, 1.0],'n_estimators':[10,20,60,100]}}
                        reg_dict.update(GBR_dict)
                    elif(i=="AdaBoostClassifier"):
                        #append AdaBoostRegressor dictionary to main dictionary
                        ABR_dict = {'AdaBoostClassifier' : { 'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],'n_estimators':[10,20,60,100]}}
                        reg_dict.update(ABR_dict)
                    else:
                        print("no algo selected")

        self.main_dict = reg_dict
        #take the values for the constructor
        top = tk.Toplevel(self.root, width=450, height=540)
        if(len(mess_age)):
            mess_age = "This is suggested algorithm : "+mess_age
        else:
            mess_age = "" 
        tk.Label(top, text=mess_age).place(x=80, y=20)
        tk.Label(top, text="Score").place(x=40, y=80)
        tk.Label(top, text="0-reg, 1-classi").place(x=40, y=140)
        tk.Label(top, text="TimeMode low/high").place(x=40, y=200)
        tk.Label(top, text="CV").place(x=40, y=260)
        tk.Label(top, text="Seed").place(x=40, y=320)
        tk.Label(top, text="Itterations").place(x=40, y=380)
        tk.Label(top, text="N_Jobs").place(x=40, y=440)
        self.score
        self.rcflag.set("False")
        self.utime_mode.set("low")
        self.cv.set("3")
        self.seed.set("100")
        self.iterr.set("10")
        self.n_jobs.set("-1")
        tk.Entry(top, font=('arial', 17), textvariable=self.score).place(x=120, y=80, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.rcflag).place(x=120, y=140, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.utime_mode).place(x=120, y=200, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.cv).place(x=120, y=260, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.seed).place(x=120, y=320, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.iterr).place(x=120, y=380, width=230)
        tk.Entry(top, font=('arial', 17), textvariable=self.n_jobs).place(x=120, y=440, width=230)

        tk.Button(top, text='Continue', bg='#003171', fg='white', font=('arial', 12), command=self.call_init).place(x=60, y=500, width=100)
        tk.Button(top, text='Close', bg='#003171', fg='white', font=('arial', 12), command=top.destroy).place(x=180, y=500, width=100)
        
        print("value : ",bool(self.rcflag.get()))
        print("type is : ",type(bool(self.rcflag.get())))

        print("value : ",self.utime_mode.get())
        print("type is : ",type((self.score.get())))

        print("value : ",(self.score.get()))
        print("type is : ",type(self.utime_mode.get()))

        print("value : ",int(self.cv.get()))
        print("type is : ",type(int(self.cv.get())))

        print("value : ",int(self.iterr.get()))
        print("type is : ",type(int(self.iterr.get())))

        print("This VALU FOUND : ",self.x_dataframe)
        #print("type is : ",)

        print("This FOUND : ",self.y_dataframe)
        #print("type is : ",type(bool(self.rcflag.get())))
        print("this is end of file ")

    def call_init(self):
        
        # print("value : ",bool(self.rcflag.get()))
        # print("type is : ",type(bool(self.rcflag.get())))

        # print("value : ",self.utime_mode.get())
        # print("type is : ",type((self.score.get())))

        # print("value : ",(self.score.get()))
        # print("type is : ",type(self.utime_mode.get()))

        # print("value : ",int(self.cv.get()))
        # print("type is : ",type(int(self.cv.get())))

        # print("value : ",int(self.iterr.get()))
        # print("type is : ",type(int(self.iterr.get())))

        # print("This VALU FOUND : ",self.x_dataframe)
        # #print("type is : ",)

        # print("This FOUND : ",self.y_dataframe)
        # #print("type is : ",type(bool(self.rcflag.get())))

        maj1_obj = maj1.SVLearningCore(self.x_dataframe,self.y_dataframe,self.main_dict,int(self.score.get()),bool(self.rcflag.get()),self.utime_mode.get(),int(self.cv.get()),int(self.seed.get()),int(self.iterr.get()),int(self.n_jobs.get()))
        self.trained_model = maj1_obj.startlearning()[1]
        print("THE FINAL MODEL IS ", self.trained_model)
        op = g_gen.VisualLearningCore(self.trained_model, self.x_dataframe, self.y_dataframe)
        op.all_encoder()

        #call the constructor of major0_1 and save it in object
        #call the startLearning from that object

    def open_console_window(self):
        top = tk.Toplevel(self.root, width=500, height=500)  
        console_frame = tk.Frame(top, bg='lavender')
        console_frame.place(x=15, y=15, width=470, height=370)
        self.textbox = tk.Text(console_frame, bg='black', fg='white')
        # self.textbox.config(state='readonly')
        self.textbox.pack(expand=True, fill='both')

        pl = PrintLogger(self.textbox)

        # replace sys.stdout with our object
        sys.stdout = pl
        self.print_()

    def print_(self):
        for i in range(11):
            print(i,'The is on console')

    def create_algo_list(self, algo):
        if algo not in self.algo_list:
            self.algo_list.append(algo)
        else:
            self.algo_list.remove(algo)
        print(self.algo_list)

    def items_selected(self, event):
        """ handle item selected event
        """
        # get selected indices
        self.selected_indices = self.listbox.curselection()
        # get selected items
        self.selected_langs = [self.listbox.get(i) for i in self.selected_indices]
        msg = f'You selected: {self.selected_langs}'

        temp_clean = clean.dataprocessing(self.filename)
        self.x_dataframe = temp_clean.get_x(self.selected_langs, self.filename)
        print("=============================",self.x_dataframe)

    def upload_file(self):
        file = filedialog.askopenfile()
        self.filename = file.name.split('/')[-1]
        shutil.copy(file.name, os.getcwd())
        
        if file != None:
            if file.name.split('.')[1] == 'xlsx':
                self.data = pd.read_excel(file.name.split('/')[-1])
            elif file.name.split('.')[1] == 'csv':
                self.data = pd.read_csv(file.name.split('/')[-1])
            self.file.config(text=file.name.split('/')[-1])
        else:    
            showerror(title='Error', message='Please select any file!')

            

root = tk.Tk()
ob = Data(root)
tk.mainloop()