# Import Tkinter library
from tkinter import *



# Create an instance of tkinter frame
root = Tk()

# Set the geometry of Tkinter frame
root.geometry("700x700")
function=[]
inputs=[]
flages=[]
# Define Function to print the input value
#features
def display_input():

      ##learning rate
      inputs.append(e.get())
      ##### input 2 epoch
      inputs.append(e2.get())
      ##### input 3 N hidden layer
      inputs.append(e3.get())
      ##### input 4 N neural in hiddden layers
      inputs.append(e4.get())
      print('inputs:',inputs)
      ### bias
      flages.append(YES.get())
      print("flages", flages)
      ## activation function
      function.append(function1.get())
      function.append(function2.get())
      print(function)


# Define empty variables



#learningrate
e=Entry(root)
e.place(x=10,y=10)
e.focus_set()

# Label
l4 = Label(root, text="LearningRate")
l4.place(x=150,y=10)

#epochs
e2=Entry(root)
e2.place(x=10,y=40)
e2.focus_set()

l5 = Label(root, text="Epochs")
l5.place(x=150,y=40)

## #hidden layers
e3=Entry(root)
e3.place(x=250,y=10)
e3.focus_set()

# Label
l6 = Label(root, text="Number of hidden layers")
l6.place(x=350,y=10)
## #number of neurons in each hidden layer
e4=Entry(root)
e4.place(x=250,y=40)
e4.focus_set()

# Label
l7 = Label(root, text="number of neurons in each hidden layer")
l7.place(x=350,y=40)


#baise
#Define empty variables
YES = IntVar()
NO = IntVar()
#
# # Label
l8 = Label(root, text="Do you want bais ?")
l8.place(x=250,y=60)
# Define a Checkbox
c1 = Checkbutton(root, text="baise", variable=YES, onvalue=1, offvalue=0)
c1.deselect()
c1.place(x=250,y=80)

#### activation function
function1 = StringVar()
function2 = StringVar()
l9 = Label(root, text="Choose  one function")
l9.place(x=10,y=60)
c1 = Checkbutton(root, text="Sigmoid function", variable=function1, onvalue='Sigmoid function', offvalue='no')
c1.deselect()
c1.place(x=10,y=80)

c2 = Checkbutton(root, text="Hyperbolic Tangent ", variable=function2, onvalue='Hyperbolic Tangent', offvalue='no')
c2.deselect()
c2.place(x=10,y=100)

# button
button1 = Button(root, text='finsh', command=lambda: display_input())
button1.place(x=250,y=120)
root.mainloop()

