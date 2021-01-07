from tkinter import *
from PIL import ImageTk
from AttendenceProject import dec2
from AttendenceProject import takeSS
from AttendenceProject import record
from AttendenceProject import atend



class Login_System:
    def __init__(self, root):
        # print("hello")
        self.root = root
        self.root.title("Attendance System")
        self.root.geometry("2560x1600+0+1")

        #Variables------------

        self.uname = StringVar()
        self.pass_ = StringVar()


        #=========All Images==============
        self.bg_icon = ImageTk.PhotoImage(file="Images/apple1.jpg")
        self.user_icon = ImageTk.PhotoImage(file="Images/friends.png")
        self.pass_icon = ImageTk.PhotoImage(file="Images/key-chain.png")
        self.logo_icon = ImageTk.PhotoImage(file="Images/acc.png")
        self.rec_icon = ImageTk.PhotoImage(file="Images/rec.png")

        bg_lbl = Label(self.root, image=self.bg_icon).pack()

        title = Label(self.root, text="Attendance System", font=("times new roman", 35, "bold"), bg="aqua", fg="indigo", bd=10, relief=GROOVE)
        title.place(x=0, y=0, relwidth = 1)

        Login_Frame = Frame(self.root, bg="black")
        # , bg = 'white')
        Login_Frame.place(x=557,y=320)

        logolbl=Label(Login_Frame, image=self.logo_icon, bg='black',compound=CENTER, bd=0).grid(row=0, columnspan=2,pady=0)

        lbluser=Label(Login_Frame, text="Register of the Class", image=self.user_icon,fg="sky blue", compound=LEFT, font=("times new roman", 30, "bold"), bg="black").grid(row=1, column=0, padx=0, pady=0)
        # txtuser=Entry(Login_Frame,bd=2,textvariable=self.uname, relief=GROOVE,font=("",10)).grid(row=1,column=1,padx=0)

        # lblpass = Label(Login_Frame, text="Password", image=self.pass_icon, compound=LEFT, font=("times new roman", 30, "bold"), bg="black", fg="white").grid(row=2, column=0, padx=0, pady=0)
        # txtpass = Entry(Login_Frame,bd=2,textvariable=self.pass_,relief=GROOVE,font=("",10)).grid(row=2,column=1,padx=0)



        btn_log = Button(Login_Frame, text="Press to take Attendance", width=30, font=("times new roman", 20, "bold"),
                         fg="green", command=dec2, bg="black")
        btn_log.grid(row=3, columnspan=1, pady=10)

        ss_btn = Button(Login_Frame, text="Take the Screenshot", width=30, font=("times new roman", 20, "bold"),
                         fg="green", command=takeSS, bg="black")
        ss_btn.grid(row=4, columnspan=1, pady=10)

        rc_btn = Button(Login_Frame, text="Take the Screenshot", height= 30, image=self.rec_icon,
                        fg="green", command=record, bg="black")
        rc_btn.grid(row=5, columnspan=1, pady=10, padx=0)


root = Tk()
obj = Login_System(root)
root.mainloop()





