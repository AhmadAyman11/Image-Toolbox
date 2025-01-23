# Imports all libraries needed
import statistics ,cv2 ,os ,re ,random ,pytesseract 
from tkinter import DISABLED, HORIZONTAL, NORMAL, NW, Entry, Button, Canvas, Label, Scale, StringVar, Text, messagebox , Tk, filedialog, PhotoImage, RAISED, Toplevel, filedialog, Text, filedialog, WORD, BOTH, END
import tkinter as tk
from tkinter.font import BOLD
from tkinter.tix import *
import PIL.Image, PIL.ImageTk
from PIL import Image, ImageTk
from tkinter.messagebox import showinfo
import tkinter.messagebox as messagebox
from tkinter.ttk import Combobox
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Part of making functions

# ////////////////////////////////////////////Top////////////////////////////////////////////

# //////////////////////Upload Image//////////////////////
def uploadImage():
    global image
    global photo
    global orignalImg
    global finalEdit
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select image", filetypes=(("JPG File","*.jpg"),("PNG File","*.png"),("All Files","*.*")))
    image = PIL.Image.open(fln)
    image = asarray(image)
    if image.shape[0] > 700 or image.shape[1] > 600:
        dim = (700,600)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    orignalImg = image
    finalEdit = image
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Save Image//////////////////////
def saveC(path, image, jpg_quality=None, png_compression=None):
    if jpg_quality:
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    elif png_compression:
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    else:
        cv2.imwrite(path, image)

def Saving():
    global finalEdit
    global image
    global photo
    finalEdit = image
    cv2.imwrite('Edit.jpg',finalEdit)
    messagebox.showinfo(title="Saving process", message="Saving Done Correctly")
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Reset Image//////////////////////
def resets():
    global image
    global orignalImg
    global finalEdit
    global photo

    # Reset the image to the original image stored in 'orignalImg'
    image = np.copy(orignalImg)  # Use np.copy to ensure a new copy of the original image

    # Reset the finalEdit as well
    finalEdit = np.copy(orignalImg)

    # Convert the image to a PhotoImage object for display
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))

    # Reset the canvas size to the original image's dimensions
    canvas.config(width=orignalImg.shape[1], height=orignalImg.shape[0])

    # Display the original image on the canvas
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)


# //////////////////////Restore Image//////////////////////
def restoreBtn():
    global image
    global photo
    finalEdit = cv2.imread("Edit.jpg",0)
    image = finalEdit
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Crop Image//////////////////////
def cropImage():
    global canvas, image, photo, finalEdit

    if image is None:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return

    cropping = [False]
    rect_coords = [0, 0, 0, 0]
    rect_id = [None]

    def start_crop(event):
        """Start defining the crop area when the mouse is pressed."""
        global image  # Declare the variable as global
        rect_coords[0], rect_coords[1] = event.x, event.y
        cropping[0] = True
        if rect_id[0]:
            canvas.delete(rect_id[0])  # Delete any existing rectangle

    def update_crop(event):
        """Update the rectangle while dragging the mouse."""
        if cropping[0]:
            if rect_id[0]:
                canvas.delete(rect_id[0])  # Delete the previous rectangle
            rect_coords[2], rect_coords[3] = event.x, event.y
            rect_id[0] = canvas.create_rectangle(
                rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3],
                outline="red", width=2
            )

    def finish_crop(event):
        """Finish defining the crop area and perform cropping when the mouse is released."""
        global image, photo, finalEdit  # Declare variables as global
        if cropping[0]:
            cropping[0] = False
            x1, y1, x2, y2 = (
                min(rect_coords[0], rect_coords[2]),
                min(rect_coords[1], rect_coords[3]),
                max(rect_coords[0], rect_coords[2]),
                max(rect_coords[1], rect_coords[3]),
            )
            canvas.delete(rect_id[0])  # Remove the rectangle from the canvas

            # Convert canvas coordinates to image coordinates
            img_h, img_w = image.shape[:2]
            canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()

            crop_x1 = int(x1 * img_w / canvas_w)
            crop_y1 = int(y1 * img_h / canvas_h)
            crop_x2 = int(x2 * img_w / canvas_w)
            crop_y2 = int(y2 * img_h / canvas_h)

            # Validate the crop area
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(img_w, crop_x2), min(img_h, crop_y2)

            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                messagebox.showwarning("Invalid Crop Area", "The cropping area is invalid.")
                return

            # Perform cropping
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Update global variables
            image = cropped_image
            finalEdit = cropped_image
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cropped_image))

            # Update the canvas display
            canvas.config(width=cropped_image.shape[1], height=cropped_image.shape[0])
            canvas.create_image(0, 0, image=photo, anchor=NW)

            messagebox.showinfo("Croping", "Image cropped successfully!")

    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", start_crop)
    canvas.bind("<B1-Motion>", update_crop)
    canvas.bind("<ButtonRelease-1>", finish_crop)


# ////////////////////////////////////////////EDIT////////////////////////////////////////////



# //////////////////////Flip Image Horizontal//////////////////////
def flipingHeFunction():
    global image
    global photo
    # Flip image horizontally using slicing
    image = np.fliplr(image)  # Flip Left-Right
    
    # Update photo and display on canvas
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Flip Image Vertical//////////////////////
def flipingVeFunction():
    global image
    global photo
    # Flip image vertically using slicing
    image = np.flipud(image)  # Flip Up-Down
    
    # Update photo and display on canvas
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Rotate Image//////////////////////
def rotBtn():
    global image
    global photo
    image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Translate Image UP//////////////////////
def trUpBtn():
    global image
    global photo
    M = np.float32([[1,0,0],[0,1,10]])
    image =cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0 , 0 ,image=photo ,anchor=NW)

# //////////////////////Translate Image Down//////////////////////
def trDwonBtn():
    global image
    global photo
    M = np.float32([[1,0,0],[0,1,-10]])
    image =cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0 , 0 ,image=photo ,anchor=NW)

# //////////////////////Translate Image Left//////////////////////
def Tleft():
    global image
    global photo
    M = np.float32([[1, 0, 10],[0, 1, 0]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Translate Image Right//////////////////////
def Tright():
    global image
    global photo
    M = np.float32([[1, 0, -10],[0, 1, 0]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)    

# //////////////////////Image Details//////////////////////
def details():
    global image
    global photo
    print("img", image)
    smallest = np.amin(image)
    biggest = np.amax(image)
    print("smallest", smallest)
    print("biggest", biggest)
    avarage = np.average(image)
    print("avarage", avarage)

    image = asarray(image)
    orignalImg = image
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Affine transformation(skew)//////////////////////
def affineTransform(val):
    global image
    global photo
    point_1 = np.float32([[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]]) 
    point_2 = np.float32([[val, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]])
    M = cv2.getAffineTransform(point_1, point_2)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Negative//////////////////////
def NegativeFunction():
    global image
    global photo

    # Apply negative effect on the image (invert pixel values)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            image[row, col] = 255 - image[row, col]

    # Update the displayed image on the canvas
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


# //////////////////////Histogram Equalization//////////////////////
def equalizeHist():
    global image
    global photo
    if(len(image.shape)<3):
        image = cv2.equalizeHist(image)    
    elif len(image.shape)==3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
      image[:,:,0] = cv2.equalizeHist(image[:,:,0])
      image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Zoom Out//////////////////////
def small():
    global image
    global photo
    image=cv2.resize(image,None,fx=0.5,fy=0.5)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Zoom IN//////////////////////
def large():
    global image
    global photo
    image=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Increase Brightness//////////////////////
def add():
    global image
    global photo
    M = np.ones(image.shape, dtype="uint8") * 100
    image=cv2.add(image,M)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Decrease Brightness//////////////////////
def sub():
    global image
    global photo
    M = np.ones(image.shape, dtype="uint8") * 100
    image=cv2.subtract(image,M)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)



# ////////////////////////////////////////////Edge Detection////////////////////////////////////////////



# //////////////////////Sobel(Horizontal)//////////////////////
def edgeDetectHor():
    global image
    global photo
    kernal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    image = cv2.filter2D(image, -1, kernal)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Sobel(Vertical)//////////////////////    
def edgeDetectVer():
    global image
    global photo
    kernal = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    image = cv2.filter2D(image, -1, kernal)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


# //////////////////////Sobel(Full)//////////////////////
def FullsobelFun():
    global image
    global photo
    kernal_1 = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])  # Sobel-X (Vertical)
    
    kernal_2 = np.array([[1,  2,  1],
                         [0,  0,  0],
                         [-1, -2, -1]])  # Sobel-Y (Horizontal)
    
    kernal_3 = np.array([[ 2,  1,  0],
                         [ 1,  0, -1],
                         [ 0, -1, -2]])  # Custom diagonal kernel

    vertical = cv2.filter2D(image, cv2.CV_64F, kernal_1)
    horizontal = cv2.filter2D(image, cv2.CV_64F, kernal_2)
    diagonal = cv2.filter2D(image, cv2.CV_64F, kernal_3)
    combined = np.sqrt(np.square(vertical) + np.square(horizontal) + np.square(diagonal))
    combined = np.uint8(255 * combined / np.max(combined))
    
    image = combined
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)



# ////////////////////////////////////////////Bottom////////////////////////////////////////////



# //////////////////////Merge Images//////////////////////
def mergeingImages():
    global photo2
    global image
    global photo
    arr = np.zeros((image.shape[0],image.shape[1]))
    fln2 = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select image", filetypes=(("JPG File","*.jpg"),("PNG File","*.png"),("All Files","*.*")))
    image2 = PIL.Image.open(fln2)
    image2 = asarray(image2)
    if image2.shape[0] > 50 or image2.shape[1] > 50:
        dim2 = (50,50)
        image2 = cv2.resize(image2,dim2, interpolation = cv2.INTER_AREA)
    photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image2))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    dim = (image.shape[1],image.shape[0])
    image2 = cv2.resize(image2,dim, interpolation = cv2.INTER_AREA)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            arr[row,col] = image[row,col] * 0.8 + image2[row,col] * 0.2
    image = arr
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Histogram//////////////////////
def hisPlot():
    global image
    plt.hist(image.ravel(),256,[0,256],color ='tab:green')
    plt.show()  

# //////////////////////Add Noise//////////////////////
def add_salt_pepper_noise(image, probability=0.01):
    noisy_image = image.copy()
    total_pixels = noisy_image.size

    # Add salt (white) and pepper (black) noise
    num_salt = int(probability * total_pixels / 2)
    num_pepper = int(probability * total_pixels / 2)

    # Add salt noise
    salt_coords = [random.randint(0, noisy_image.shape[0] - 1), random.randint(0, noisy_image.shape[1] - 1)]
    for _ in range(num_salt):
        x = random.randint(0, noisy_image.shape[0] - 1)
        y = random.randint(0, noisy_image.shape[1] - 1)
        noisy_image[x, y] = 255  # Salt (white)

    # Add pepper noise
    for _ in range(num_pepper):
        x = random.randint(0, noisy_image.shape[0] - 1)
        y = random.randint(0, noisy_image.shape[1] - 1)
        noisy_image[x, y] = 0  # Pepper (black)

    return noisy_image

def applySaltPepperNoise():
    global image
    global photo
    probability = 0.01  # 1% of the pixels will be noise
    image = add_salt_pepper_noise(image, probability)

    # Convert the noisy image back to a format that can be shown in Tkinter
    photo = PIL.ImageTk.PhotoImage(image=Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////OCR//////////////////////
def ocr_and_display_text(root, image):
    try:
        # Perform OCR on the image
        extracted_text = pytesseract.image_to_string(image)

        # Create a new window to display the extracted text
        text_window = Toplevel(root)
        text_window.title("Extracted Text")

        # Add a text widget to show the OCR results
        text_area = Text(text_window, wrap=WORD, font=("Arial", 12))
        text_area.pack(expand=True, fill="both", padx=10, pady=10)
        text_area.insert(END, extracted_text)
        text_area.config(state="disabled")  # Make the text area read-only

        # Function to save the extracted text to the project folder
        def save_text_to_file():
            try:
                # Define the file path in the project folder
                project_path = os.getcwd()  # Get the current working directory
                file_path = os.path.join(project_path, "Extracted_Text.txt")
                
                # Save the text to the specified file
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(extracted_text)
                print(f"Text saved successfully to {file_path}")
                messagebox.showinfo(title="Saving Text", message=f"Text File Saved Successfully" + "\n" + "File saved in the same path as the project.")
            except Exception as e:
                print(f"Error saving text: {e}")
                messagebox.showerror(title="Error", message=f"Error saving text: {e}")

        # Add a Save button to the window
        save_button = Button(text_window, text="Save Text", command=save_text_to_file)
        save_button.pack(pady=10)

    except Exception as e:
        # Handle errors (e.g., OCR issues)
        print(f"Error during OCR: {e}")
        messagebox.showerror(title="Error", message=f"Error during OCR: {e}")


def open_image_and_extract_text():
    if image is not None:
        ocr_and_display_text(test, image)


# //////////////////////Compression//////////////////////
def dctCompression(block_size=8):
    global image
    try:
        # Input image
        my_string = image
        shape = my_string.shape
        print("Input image shape:", shape)

        # Check if image is grayscale or color
        if len(shape) == 3:  # Color image
            height, width, channels = shape
            compressed_image = np.zeros_like(my_string, dtype=np.uint8)
            for channel in range(channels):
                compressed_image[:, :, channel] = compress_channel(my_string[:, :, channel], block_size)
        else:  # Grayscale image
            compressed_image = compress_channel(my_string, block_size)

        # Save the compressed image
        cv2.imwrite("Compressed_DCT.jpg", compressed_image)
        print("DCT Compression successful. Compressed image saved as 'Compressed_DCT.jpg'.")
        messagebox.showinfo(title="Compression Message", message="DCT Compression Done" + "\n" + "Image saved in the same path as the project.")
    except Exception as e:
        print(f"An error occurred during DCT compression: {e}")
        messagebox.showerror(title="Error", message=f"An error occurred: {e}")

def compress_channel(channel, block_size=8, quality=50):
    # Dynamically generate a quantization matrix scaled to the block size
    quant_matrix = np.ones((block_size, block_size), dtype=np.float32) * 16
    for i in range(block_size):
        for j in range(block_size):
            quant_matrix[i, j] = 1 + (1 + i + j) * quality / block_size

    # Adjust quantization matrix based on quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    quant_matrix = np.floor((quant_matrix * scale + 50) / 100).astype(np.int32)
    quant_matrix[quant_matrix == 0] = 1  # Avoid division by zero

    height, width = channel.shape

    # Pad the channel if dimensions are not divisible by block size
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_channel = np.pad(channel, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=128)

    compressed_channel = np.zeros_like(padded_channel, dtype=np.uint8)

    # Perform block-wise DCT and quantization
    for i in range(0, padded_channel.shape[0], block_size):
        for j in range(0, padded_channel.shape[1], block_size):
            # Extract block
            block = padded_channel[i:i+block_size, j:j+block_size]

            # Apply DCT
            dct_block = cv2.dct(np.float32(block) - 128)

            # Quantize the DCT coefficients
            quantized_block = np.round(dct_block / quant_matrix) * quant_matrix

            # Apply inverse DCT
            idct_block = cv2.idct(quantized_block) + 128

            # Clip values to valid range and write back to compressed channel
            compressed_channel[i:i+block_size, j:j+block_size] = np.clip(idct_block, 0, 255)

    # Remove padding before returning the compressed image
    return compressed_channel[:height, :width]



# //////////////////////Compression(Huffman)//////////////////////
def huffmanfun():
    global image
    my_string = image
    shape = my_string.shape
    a = my_string
    print ("Entered string is:",my_string)
    my_string = str(my_string.tolist())  

    letters = []
    only_letters = []
    for letter in my_string:
        if letter not in letters:
            frequency = my_string.count(letter)
            letters.append(frequency)
            letters.append(letter)
            only_letters.append(letter)

    nodes = []
    while len(letters) > 0:
        nodes.append(letters[0:2])
        letters = letters[2:]
    nodes.sort()
    huffman_tree = []
    huffman_tree.append(nodes)

    def combine_nodes(nodes):
        pos = 0
        newnode = []
        if len(nodes) > 1:
            nodes.sort()
            nodes[pos].append("1")                       # assigning values 1 and 0
            nodes[pos+1].append("0")
            combined_node1 = (nodes[pos] [0] + nodes[pos+1] [0])
            combined_node2 = (nodes[pos] [1] + nodes[pos+1] [1])  # combining the nodes to generate pathways
            newnode.append(combined_node1)
            newnode.append(combined_node2)
            newnodes=[]
            newnodes.append(newnode)
            newnodes = newnodes + nodes[2:]
            nodes = newnodes
            huffman_tree.append(nodes)
            combine_nodes(nodes)
        return huffman_tree                                     # huffman tree generation

    newnodes = combine_nodes(nodes)

    huffman_tree.sort(reverse = True)

    checklist = []
    for level in huffman_tree:
        for node in level:
            if node not in checklist:
                checklist.append(node)
            else:
                level.remove(node)
    count = 0
    for level in huffman_tree:
        count+=1

    letter_binary = []
    if len(only_letters) == 1:
        lettercode = [only_letters[0], "0"]
        letter_binary.append(lettercode*len(my_string))
    else:
        for letter in only_letters:
            code =""
            for node in checklist:
                if len (node)>2 and letter in node[1]:           #genrating binary code
                    code = code + node[2]
            lettercode =[letter,code]
            letter_binary.append(lettercode)
            
    for letter in letter_binary:
        print(letter[0], letter[1])

    bitstring =""
    for character in my_string:
        for item in letter_binary:
            if character in item:
                bitstring = bitstring + item[1]
    binary ="0b"+bitstring

    uncompressed_file_size = len(my_string)*7
    compressed_file_size = len(binary)-2
    print("Your original file size was", uncompressed_file_size,"bits. The compressed size is:",compressed_file_size)
    print("This is a saving of ",uncompressed_file_size-compressed_file_size,"bits")
    output = open("compressed.txt","w+")
    print("Compressed file generated as compressed.txt")
    output = open("compressed.txt","w+")
    print("Decoding.......")
    output.write(bitstring)

    bitstring = str(binary[2:])
    uncompressed_string =""
    code =""
    for digit in bitstring:
        code = code+digit
        pos=0                                        #iterating and decoding
        for letter in letter_binary:
            if code ==letter[1]:
                uncompressed_string=uncompressed_string+letter_binary[pos] [0]
                code=""
            pos+=1

    print("Your UNCOMPRESSED data is:")

    temp = re.findall(r'\d+', uncompressed_string)
    res = list(map(int, temp))
    res = np.array(res)
    res = res.astype(np.uint8)
    res = np.reshape(res, shape)
    cv2.imwrite("CompressHoffman.jpg",res)
    if a.all() == res.all():
        print("Success")
        messagebox.showinfo(title="compressor Message", message="compressor Done"+"\n"+"image store in same path of project")

# //////////////////////Erosion Function//////////////////////
def apply_erosion(kernel_size=(5, 5), iterations=1):
    """Apply erosion to the currently edited image."""
    global finalEdit
    global photo

    if finalEdit is None:
        messagebox.showwarning(title="Warning", message="No image to apply erosion. Please upload an image first.")
        return

    # Create the kernel and apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_image = cv2.erode(finalEdit, kernel, iterations=iterations)
    
    # Update the final edited image and display it
    finalEdit = eroded_image
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(finalEdit))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Dilation Function//////////////////////
def apply_dilation(kernel_size=(5, 5), iterations=1):
    """Apply dilation to the currently edited image."""
    global finalEdit
    global photo

    if finalEdit is None:
        messagebox.showwarning(title="Warning", message="No image to apply dilation. Please upload an image first.")
        return

    # Create the kernel and apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated_image = cv2.dilate(finalEdit, kernel, iterations=iterations)
    
    # Update the final edited image and display it
    finalEdit = dilated_image
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(finalEdit))
    canvas.create_image(0, 0, image=photo, anchor=NW)




# ////////////////////////////////////////////Filters////////////////////////////////////////////



# //////////////////////Choose Filter//////////////////////
def FilterCombobox(event):
    global image
    global photo
    filters = event.widget.get()

    # Reset motion blur scale value when switching filters
    motion_blur_scale.set(0)  # Reset to default value

    try:
        if filters == 'Gray-scale':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filters == 'Averaging Filter':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            kernel = np.ones((3, 3), np.float32) / 9
            image = cv2.filter2D(image, -1, kernel)
        elif filters == 'Cone Filter':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            kernel = np.ones((5, 5), np.float32) / 25
            image = cv2.filter2D(image, -1, kernel)
        elif filters == 'None':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            image = cv2.filter2D(image, -1, kernel)
        elif filters == 'Motion-blur':
            motion_blur_scale['state'] = NORMAL
            disableShearingBtn()
        elif filters == 'sharping':
            motion_blur_scale['state'] = DISABLED
            enableShearingBtns()
        elif filters == 'Gaussian blur':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            image = cv2.GaussianBlur(image, (9, 9), 0)
        elif filters == 'Median blur':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            image = cv2.medianBlur(image, 9)
        elif filters == 'Bilateral filter':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            image = cv2.bilateralFilter(image, 10, 25, 25)
        elif filters == 'Circular Filter':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            kernel = create_circular_kernel(5)
            image = cv2.filter2D(image, -1, kernel)
        elif filters == 'Pyramidal Filter':
            motion_blur_scale['state'] = DISABLED
            disableShearingBtn()
            kernel = create_pyramidal_kernel(5)
            image = cv2.filter2D(image, -1, kernel)

        # Convert image to displayable format
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)
    except Exception as e:
        print(f"An error occurred: {e}")

def disableShearingBtn():
    searingBt1['state'] = DISABLED
    searingBt2['state'] = DISABLED
    searingBt3['state'] = DISABLED

def enableShearingBtns():
    searingBt1['state'] = NORMAL
    searingBt2['state'] = NORMAL
    searingBt3['state'] = NORMAL

def create_circular_kernel(size):
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= center**2
    kernel = np.zeros((size, size), np.float32)
    kernel[mask] = 1
    return kernel / kernel.sum()

def create_pyramidal_kernel(size):
    center = size // 2
    kernel = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = max(0, center - max(abs(i - center), abs(j - center)) + 1)
    return kernel / kernel.sum()



# //////////////////////Motion Blur//////////////////////
def scaleOfMotion(val):
    global image
    global photo
    size = motion_blur_scale.get()
    kernal_motion_blur = np.zeros((size,size))
    kernal_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernal_motion_blur = kernal_motion_blur/size
    image = cv2.filter2D(image, -1, kernal_motion_blur)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Power Transformation//////////////////////
def powerTransformation():
    global image
    global photo
    try:
        gamma_value = float(gammavalueInput.get())
        if gamma_value < 0 or gamma_value > 3:
            messagebox.showwarning("Invalid Input", "Gamma value must be between 0 and 3")
            return
    except ValueError:
            messagebox.showwarning("Invalid Input", "Invalid gamma value entered")
            return

    # Perform the gamma correction
    arr = np.array(255 * (image / 255) ** gamma_value, dtype='uint8')

    cv2.normalize(arr, arr, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(arr, arr)
    image = arr

    # Update the image on the canvas
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


# //////////////////////Bit Plane//////////////////////
def bitPlaneCombobox(even):
    global image
    global photo
    val = even.widget.get()
    bit_positions = {
        '8': 7,
        '7': 6,
        '6': 5,
        '5': 4,
        '4': 3,
        '3': 2,
        '2': 1,
        '1': 0
    }
    bit_position = bit_positions.get(val, None)
    if bit_position is not None:
        image = (image >> bit_position) & 1
        image = image * 255
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Threshold//////////////////////
def thresholdBtn():
    global finalEdit
    global image
    global photo
    if(minInput.get() == '' or maxInput.get() == ''):
        messagebox.showwarning(title="Warning", message="Enter Min Value or Max Value of threshold")
    elif(float(maxInput.get()) <= float(minInput.get())):
        messagebox.showwarning(title="Warning", message="The max Threshold must bigger then min threshold")
    else:
        ret1, thresh1 = cv2.threshold(image, float(minInput.get()), float(maxInput.get()), cv2.THRESH_BINARY_INV)
        image = thresh1
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Canny Edge Detection//////////////////////
def edgeDetectCanny():
    global image
    global photo
    if(minInput.get() == '' or maxInput.get() == ''):
        messagebox.showwarning(title="Warning", message="Enter Min Value or Max Value of threshold")
    elif(float(maxInput.get()) <= float(minInput.get())):
        messagebox.showwarning(title="Warning", message="The max Threshold must bigger then min threshold")
    else:
        image = cv2.Canny(image, float(minInput.get()), float(maxInput.get()))
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Gray Level Slicing//////////////////////
def grayMethod():
    global image
    global photo

    # Ensure the image is grayscale
    if len(image.shape) == 3:  # If the image is not already grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Automatic mode: Use mean and median
    if minInputGray.get() == '' or maxInputGray.get() == '':
        m = statistics.mean(gray_image.ravel())
        med = statistics.median(gray_image.ravel())
        # Apply thresholding condition
        mask = (gray_image > med) & (gray_image < m)
        gray_image[mask] = 255

    # Manual mode: Use user-provided thresholds
    else:
        try:
            min_gray = int(minInputGray.get())
            max_gray = int(maxInputGray.get())
            if min_gray >= max_gray:
                messagebox.showwarning(title="Warning", message="The max gray level must be greater than the min gray level")
                return
        except ValueError:
            messagebox.showwarning(title="Warning", message="Invalid input for gray level thresholds")
            return

        # Apply thresholding condition
        mask = (gray_image > min_gray) & (gray_image < max_gray)
        gray_image[mask] = 255

    # Update the global image and display it
    image = gray_image.copy()
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(gray_image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Low Pass Filter//////////////////////
def lowFreq():
    import numpy as np
    import PIL.Image, PIL.ImageTk
    import matplotlib.pyplot as plt
    
    global image
    global photo
    
    # Get radius from input or set default
    if frequancyFilterInput.get() == '':
        r = 60  # Default radius
    else:
        r = int(frequancyFilterInput.get())
    
    # Ensure image is grayscale
    if len(image.shape) != 2:
        messagebox.showinfo(title="Low Pass Filter", message="The Input image must be grayscale")
        return
    
    # Get image dimensions
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2  # Calculate center

    # Create low-pass filter mask
    H = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            D = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
            if D <= r:
                H[row, col] = 1

    # Apply Fourier Transform
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)

    # Display original frequency spectrum
    plt.figure(1)
    plt.title("Original Frequency Spectrum")
    plt.imshow(np.log1p(np.abs(f_shifted)), cmap='gray')
    plt.colorbar()
    plt.show()

    # Apply the filter in frequency domain
    LPF = f_shifted * H

    # Display filtered frequency spectrum
    plt.figure(2)
    plt.title("Filtered Frequency Spectrum")
    plt.imshow(np.log1p(np.abs(LPF)), cmap='gray')
    plt.colorbar()
    plt.show()

    # Perform Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(LPF)
    filtered_image = np.abs(np.fft.ifft2(f_ishift))

    # Normalize the image to 0-255
    filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 255
    filtered_image = filtered_image.astype(np.uint8)

    # Update the global image and display on the canvas
    image = filtered_image
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.delete("all")  # Clear previous images
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////High Pass Filter//////////////////////
def highFreq():
    import numpy as np
    import PIL.Image, PIL.ImageTk
    import matplotlib.pyplot as plt
    
    global image
    global photo
    
    # Get radius from input or set default
    if frequancyFilterInput.get() == '':
        r = 60  # Default radius
    else:
        r = int(frequancyFilterInput.get())
    
    # Ensure image is grayscale
    if len(image.shape) != 2:
        messagebox.showinfo(title="High Pass Filter", message="The Input image must be grayscale")
        return
    
    # Get image dimensions
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2  # Calculate center

    # Create high-pass filter mask
    H = np.ones((rows, cols), dtype=np.float32)  # Start with a mask that passes all frequencies
    for row in range(rows):
        for col in range(cols):
            D = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
            if D <= r:  # Block low frequencies by setting to 0
                H[row, col] = 0

    # Apply Fourier Transform
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)

    # Display original frequency spectrum
    plt.figure(1)
    plt.title("Original Frequency Spectrum")
    plt.imshow(np.log1p(np.abs(f_shifted)), cmap='gray')
    plt.colorbar()
    plt.show()

    # Apply the high-pass filter in the frequency domain
    HPF = f_shifted * H

    # Display filtered frequency spectrum
    plt.figure(2)
    plt.title("Filtered Frequency Spectrum")
    plt.imshow(np.log1p(np.abs(HPF)), cmap='gray')
    plt.colorbar()
    plt.show()

    # Perform Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(HPF)
    filtered_image = np.abs(np.fft.ifft2(f_ishift))

    # Normalize the image to 0-255
    filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 255
    filtered_image = filtered_image.astype(np.uint8)

    # Update the global image and display on the canvas
    image = filtered_image
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.delete("all")  # Clear previous images
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Low Sharping//////////////////////
def Sharping1():
    global image
    global photo
    kernal_shearing = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Medium Sharping//////////////////////
def Sharping2():
    global image
    global photo
    kernal_shearing = np.array([[1, 1, 1],[1, -7, 1],[1, 1, 1]])
    image = cv2.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////High Sharping//////////////////////    
def Sharping3():
    global image
    global photo
    kernal_shearing = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
    image = cv2.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# //////////////////////Logaritmtic Transformation////////////////////// 
def logBtn():
    global image
    global photo
    c = 255 / np.log( 1 + np.max(image))
    logImage = c * (np.log(image + 1))
    image = np.array(logImage, dtype=np.uint8)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


# /////////////////////////////////////////////////////Buttons/////////////////////////////////////////////////////


# //////////////////////Frame//////////////////////
test = Tk()
width= test.winfo_screenwidth()               
height= test.winfo_screenheight()               
test.geometry("%dx%d" % (width, height))
test.resizable(False , False)
test.title("ToolBox")
test.config(background='white')
test.state('zoomed')
tip= Balloon(test)
tip.config(bg='green')

tip.label.config(bg='red',fg='white',bd=2)
def changeOnHover(button):
    button.bind("<Enter>", func=lambda e: button.config(
        border=1))
    button.bind("<Leave>", func=lambda e: button.config(
        border=0))

canvas = Canvas(test , width = 700, height = 600,bg='#c5cac9')
canvas.place(x=width/4,y=height/7)

# //////////////////////Colors//////////////////////
topIconsColor="#ffffff"
bottomIconsColor="#ffffff"
leftIconColor="#ffffff"
rightIconColor="#ffffff"
textColor="#000000"
headingColor="white"


# ////////////////////////////////////////////Top////////////////////////////////////////////


canvasTop = Canvas(test , width = width, height = height/7-4,bg='#70aba0')
canvasTop.place(x=0,y=0)

l = Label(test, text = "Tool-Box")
l.config(font =("Helvetica", 22, BOLD),bg='#70aba0',fg=headingColor,pady=10)
l.pack()

browseIcon = PhotoImage(file= r"./NewIcons/photo.png")
browseBtn = Button(test,image=browseIcon,border=0 ,fg='black', bg=topIconsColor,width=50,height=40, command=uploadImage)
browseBtn.place(x=405,y=75)
tip.bind_widget(browseBtn,balloonmsg="Browse Image: That button use to browse images")
changeOnHover(browseBtn)

saveIcon = PhotoImage(file= r"./NewIcons/diskette.png")
saveBtn = Button(test, image=saveIcon,border=0, fg='black', bg=topIconsColor ,width=50,height=40, command=Saving)
saveBtn.place(x=555,y=75)
tip.bind_widget(saveBtn,balloonmsg="Save Image: That button use to Save image as edit image in same file of project"+"\n"+" and you can restore it")
changeOnHover(saveBtn)

resetIcon = PhotoImage(file= r"./NewIcons/reset.png")
resetBtn=Button(test, image=resetIcon,border=0,fg='black',bg=topIconsColor,width=50,height=40, command=resets)
resetBtn.place(x=705,y=75)
tip.bind_widget(resetBtn,balloonmsg="Reset Image: That button use to reset image that choosed")
changeOnHover(resetBtn)

restoreIcon = PhotoImage(file= r"./NewIcons/recovery.png")
restoreBtn=Button(test, image=restoreIcon,border=0,fg='black',bg=topIconsColor,width=50,height=40, command=restoreBtn)
restoreBtn.place(x=855,y=75)
tip.bind_widget(restoreBtn,balloonmsg="Restore Image: That button use to restore image that name edit image"+"\n"+"to complete work on it from last step")
changeOnHover(restoreBtn)

cropIcon = PhotoImage(file= r"./NewIcons/crop.png")
cropBtn=Button(test, image=cropIcon,border=0,fg='black',bg=topIconsColor,width=50,height=40, command=cropImage)
cropBtn.place(x=1005,y=75)
tip.bind_widget(cropBtn,balloonmsg="Crop Image: That button use to crop the image to resize it")
changeOnHover(cropBtn)

# ////////////////////////////////////////////Bottom////////////////////////////////////////////
canvasBottom = Canvas(test , width = width, height = height/7+6,bg='#70aba0')
canvasBottom.place(x=0,y=700)

dctIcon = PhotoImage(file= r"./NewIcons/folder.png")
dctBtn = Button(test, image=dctIcon,border=0, fg='black', bg=bottomIconsColor,width=50,height=40, command=dctCompression)
dctBtn.place(x=775,y=740)
tip.bind_widget(dctBtn,balloonmsg="DCT: it is a way for compress the image in a lossy way")
changeOnHover(dctBtn)

histGraphBtnIcon = PhotoImage(file= r"./NewIcons/wave-graph.png")
histGraphBtn = Button(test, image=histGraphBtnIcon, fg='black', bg=bottomIconsColor,width=50,height=40, border=0,command=hisPlot)
histGraphBtn.place(x=385,y=740)
tip.bind_widget(histGraphBtn,balloonmsg="Histogram: is a graph, which shows intensity distribution of an image "+"\n"
                                        +"mean that graph display the number of iterations for each pixel's intensity")
changeOnHover(histGraphBtn)

detectObjIcon = PhotoImage(file= r"./NewIcons/text.png")
DetectObjBtn = Button(test, image=detectObjIcon, fg='black', border=0 ,width=50,height=40, bg=bottomIconsColor, command=open_image_and_extract_text)
DetectObjBtn.place(x=645,y=740)
tip.bind_widget(DetectObjBtn,balloonmsg="OCR: use to detect text and display it")
changeOnHover(DetectObjBtn)


noiseIcon = PhotoImage(file= r"./NewIcons/equalizer.png")
noiseBtn = Button(test,  image=noiseIcon, fg='black', border=0 ,width=50,height=40, bg=bottomIconsColor, command=applySaltPepperNoise)
noiseBtn.place(x=515,y=740)
tip.bind_widget(noiseBtn,balloonmsg="Add salt-pepper Noise")
changeOnHover(noiseBtn)

mergeingImagesIcon = PhotoImage(file= r"./NewIcons/merge.png")
mergeingImagesBtn = Button(test, image=mergeingImagesIcon, fg='black', bg=bottomIconsColor,border=0,width=50,height=40, command=mergeingImages)
mergeingImagesBtn.place(x=255,y=740)
tip.bind_widget(mergeingImagesBtn,balloonmsg="Merge image: use to merge two image together")
changeOnHover(mergeingImagesBtn)

HoveMentIcon = PhotoImage(file= r"./NewIcons/compress.png")
HoveMentIconbtn = Button(test, image=HoveMentIcon, fg='black', bg=bottomIconsColor,border=0,width=50,height=40, command=huffmanfun)
HoveMentIconbtn.place(x=905,y=740)
tip.bind_widget(HoveMentIconbtn,balloonmsg="Huffman: it is a way for compress the image in a loseless way")
changeOnHover(HoveMentIconbtn)

ErosionIcon = PhotoImage(file= r"./NewIcons/rock.png")
Erosionbtn = Button(test, image=ErosionIcon, fg='black', bg=bottomIconsColor,border=0,width=50,height=40, command=apply_erosion)
Erosionbtn.place(x=1035,y=740)
tip.bind_widget(Erosionbtn,balloonmsg="Erosion: Shrinks the white regions")
changeOnHover(Erosionbtn)

DilationIcon = PhotoImage(file= r"./NewIcons/resize-expand.png")
Dilationbtn = Button(test, image=DilationIcon, fg='black', bg=bottomIconsColor,border=0,width=50,height=40, command=apply_dilation)
Dilationbtn.place(x=1165,y=740)
tip.bind_widget(Dilationbtn,balloonmsg="Dilation: Expands the white regions")
changeOnHover(Dilationbtn)

# ////////////////////////////////////////////Left////////////////////////////////////////////
canvasLeft = Canvas(test , width = 380, height = 575,bg='#EDCBD2')
canvasLeft.place(x=0,y=height/7)

label = Label(test, text="Edit", relief= RAISED,width=20,height=2,bg='#59817a',fg='white', borderwidth=0,font=("Helvetica", 10))
label.place(x=100, y=150)

Move_up = PhotoImage(file= r"./NewIcons/angle-small-down (1).png")
translateBt1 =Button(test,image=Move_up,bg=leftIconColor,width=40,height=40,border=0, command=trUpBtn)
translateBt1.place(x=100,y=250)
tip.bind_widget(translateBt1,balloonmsg="Translate Bottom: use to move image to Bottom")
changeOnHover(translateBt1)

Move_down = PhotoImage(file= r"./NewIcons/angle-small-up (2).png")
translateBt2 =Button(test,image=Move_down,fg='black',bg=leftIconColor,width=40,height=40,border=0, command=trDwonBtn)
translateBt2.place(x=100,y=200)
tip.bind_widget(translateBt2,balloonmsg="Translate Top: use to move image to top")
changeOnHover(translateBt2)

Move_left = PhotoImage(file= r"./NewIcons/angle-small-right.png")
translateBt3 =Button(test,image=Move_left,fg='black',bg=leftIconColor,width=40,height=40,border=0, command=Tleft)
translateBt3.place(x=150,y=250)
tip.bind_widget(translateBt3,balloonmsg="Translate Right: use to move image to right")
changeOnHover(translateBt3)

Move_right = PhotoImage(file= r"./NewIcons/angle-small-left.png")
translateBt4 =Button(test,image=Move_right,fg='black',bg=leftIconColor,width=40,height=40,border=0, command=Tright)
translateBt4.place(x=50,y=250)
tip.bind_widget(translateBt4,balloonmsg="Translate Left: use to move image to left")
changeOnHover(translateBt4)

translateBt5 =Scale(test, orient=HORIZONTAL, from_=1, to=20, label='Skewing',border=0, command=affineTransform,bg=leftIconColor, resolution=3, tickinterval=2, length=170, sliderlength=20, showvalue=0)
translateBt5.place(x=50,y=330)
changeOnHover(translateBt5)

flipIconH = PhotoImage(file= r"./NewIcons/flip-horizontal.png")
flibBt1 =Button(test,image=flipIconH,fg='black',width=50,height=40,border=0,bg=leftIconColor, command=flipingHeFunction)
flibBt1.place(x=300,y=200)
tip.bind_widget(flibBt1,balloonmsg="Flip Vertical: use to reverses the image on the X-axis")
changeOnHover(flibBt1)

flipIconV = PhotoImage(file= r"./NewIcons/reflect.png")
flibBt2 =Button(test,image=flipIconV,fg='black',width=50,height=40,border=0,bg=leftIconColor, command=flipingVeFunction)
flibBt2.place(x=300,y=250)
tip.bind_widget(flibBt2,balloonmsg="Flip Horizontal: use to reverses the image on the Y-axis")
changeOnHover(flibBt2)

rotateIcon = PhotoImage(file= r"./NewIcons/rotate-right.png")
rotateBt1 =Button(test,image=rotateIcon,fg='black',width=50,height=40,border=0,bg=leftIconColor, command=rotBtn)
rotateBt1.place(x=300,y=300)
tip.bind_widget(rotateBt1,balloonmsg="Rotate Image: That button use to rotate image with 90 deg"+"\n"+"with clockwise")
changeOnHover(rotateBt1)

negativeBtnIcon=PhotoImage(file= r"./NewIcons/icons8-negative-32.png")
negativeBtn = Button(test, image=negativeBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=NegativeFunction)
negativeBtn.place(x=300,y=380)
tip.bind_widget(negativeBtn,balloonmsg="Negative image: use to enhancing white or"+"\n"+" grey detail embedded in dark regions of an image")
changeOnHover(negativeBtn)

histBtnIcon=PhotoImage(file= r"./NewIcons/icons8-adjust-32.png")
histBtn = Button(test, image=histBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=equalizeHist)
histBtn.place(x=300,y=430)
tip.bind_widget(histBtn,balloonmsg="Histogram Equlization: use to applied to a dark or"+"\n"+"washed out images in order to improve image contrast")
changeOnHover(histBtn)

addBtnIcon=PhotoImage(file= r"./NewIcons/lightbulb.png")
addBtn = Button(test, image=addBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=add)
addBtn.place(x=20,y=430)
tip.bind_widget(addBtn,balloonmsg="Increase the brightness of image")
changeOnHover(addBtn)

subBtnIcon=PhotoImage(file= r"./NewIcons/light-bulb.png")
subBtn = Button(test, image=subBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=sub)
subBtn.place(x=80,y=430)
tip.bind_widget(subBtn,balloonmsg="Decrease the brightness of image")
changeOnHover(subBtn)

smallBtnIcon=PhotoImage(file= r"./NewIcons/reduce.png")
smallBtn = Button(test, image=smallBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=small)
smallBtn.place(x=140,y=430)
tip.bind_widget(smallBtn,balloonmsg="Decrease the size of the image")
changeOnHover(smallBtn)

largeBtnIcon=PhotoImage(file= r"./NewIcons/expand (1).png")
largeBtn = Button(test, image=largeBtnIcon, fg='black', bg=leftIconColor,border=0,width=50,height=40, command=large)
largeBtn.place(x=200,y=430)
tip.bind_widget(largeBtn,balloonmsg="Doubling the size of the image")
changeOnHover(largeBtn)


# ///////////////////Edge Detection//////////////////////
label = Label(test, text=" Edge Detection", relief= RAISED,width=20,height=2,bg='#59817a',fg='white', borderwidth=0,font=("Helvetica", 10))
label.place(x=100, y=500)

edgeBtn_HorIcon=PhotoImage(file= r"./NewIcons/menu.png")
edgeBtn_Hor = Button(test, image=edgeBtn_HorIcon, fg='black',width=50,height=40,border=0, bg=rightIconColor, command=edgeDetectHor)
edgeBtn_Hor.place(x=80,y=580)
tip.bind_widget(edgeBtn_Hor,balloonmsg="Sobal edge detection: use to find the edges horizental in image")
changeOnHover(edgeBtn_Hor)

edgeBtn_VerIcon=PhotoImage(file= r"./NewIcons/vertical-lines.png")
edgeBtn_Ver = Button(test,  image=edgeBtn_VerIcon ,fg='black',width=50,height=40,border=0, bg=rightIconColor, command=edgeDetectVer)
edgeBtn_Ver.place(x=155,y=580)
tip.bind_widget(edgeBtn_Ver,balloonmsg="Sobal edge detection: use to find the edges vertical in image")
changeOnHover(edgeBtn_Ver)

sobelIcon = PhotoImage(file= r"./NewIcons/hexagon.png") 
sobelBtn = Button(test, image=sobelIcon, fg='black' ,width=50,height=40,border=0,bg=rightIconColor, command=FullsobelFun)
sobelBtn.place(x=230,y=580)
tip.bind_widget(sobelBtn,balloonmsg="Sobal edge detection: use to find the edges in all direction in image")
changeOnHover(sobelBtn)

# ////////////////////////////////////////////Right////////////////////////////////////////////
canvasRight = Canvas(test , width = 500, height = 575,bg='#EDCBD2')
canvasRight.place(x=1080,y=height/7)

label = Label(test, text="Filters", relief= RAISED,width=20,fg="white",height=2,bg='#59817a', borderwidth=0,font=("Helvetica", 10))
label.place(x=1220, y=150)


label4 = Label(test, text="Choose Filter",bg='#EDCBD2',fg=textColor, relief= RAISED, borderwidth=0,border=0,font=("Helvetica", 10))
label4.place(x=1100, y=200)
filterSelect = StringVar()
filters = Combobox(test,values=('None', 'Gray-scale', 'Averaging Filter', 'Cone Filter',
                               'Motion-blur', 'Gaussian blur', 'Median blur', 'Bilateral filter',
                               'sharping', 'Circular Filter', 'Pyramidal Filter')
                   ,state='readonly',textvariable=filterSelect,width=30)
filters.place(x=1100,y=220)
filters.current(0)
filters.bind("<<ComboboxSelected>>", FilterCombobox)
motion_blur_scale = Scale(test,bg=rightIconColor, orient=HORIZONTAL, from_=1, to=20, command=scaleOfMotion, state=DISABLED, label= 'Motion Blur', resolution=3, tickinterval=2, length=170, sliderlength=20, showvalue=0)
motion_blur_scale.place(x=1330,y=220)


labelgamma = Label(test, text="Gamma Value", border=0 ,width=15,height=2,bg='#EDCBD2',fg=textColor,font=("Helvetica", 10), relief= RAISED, borderwidth=0)
labelgamma.place(x=1110, y=280)

gammavalueInput = Entry(test, width=10)
gammavalueInput.place(x=1150,y=310)

powerIcon = PhotoImage(file= r"./NewIcons/gamma.png")
powerBtn = Button(test, image=powerIcon, bg=rightIconColor, border=0,width=50,height=40, command=powerTransformation)
powerBtn.place(x=1250,y=310)
tip.bind_widget(powerBtn,balloonmsg="Power Transformation: If Gamma value < 1 the image will be more brightness,If Gamma value > 1"+"\n"
                +"the image will be more darkness, If Gamma value = 1 the image will not change")
changeOnHover(powerBtn)


label4 = Label(test, text="Bit Plane Transformation",bg='#EDCBD2',fg=textColor, relief= RAISED, borderwidth=0,border=0,font=("Helvetica", 8))
label4.place(x=1380, y=290)
bitPlaneSelect = StringVar()
bitPlane = Combobox(test,values=('None','8','7','6','5','4','3','2','1')
                   ,state='readonly',textvariable=bitPlaneSelect,width=15)
bitPlane.place(x=1380,y=310)
bitPlane.current(0)
bitPlane.bind("<<ComboboxSelected>>", bitPlaneCombobox)
tip.bind_widget(bitPlane,balloonmsg="Bit Plane Slicing: is a method of representing an image with one or more bits of the byte used for each pixel"+"\n"
                                    +"and can chooes the value of bits of the byte from combobox")


label = Label(test, text="Threshold values", relief= RAISED,width=15,height=2,bg='#EDCBD2',fg=textColor, borderwidth=0,font=("Helvetica", 10))
label.place(x=1100, y=345)

label = Label(test, text="Min", relief= RAISED,width=5,height=2,bg='#EDCBD2',fg=textColor, borderwidth=0,font=("Helvetica", 10))
label.place(x=1100, y=370)
minInput = Entry(test, width=10)
minInput.place(x=1150,y=375)

label = Label(test, text="Max", relief= RAISED,width=5,height=2,bg='#EDCBD2',fg=textColor, borderwidth=0,font=("Helvetica", 10))
label.place(x=1100, y=395)
maxInput = Entry(test, width=10)
maxInput.place(x=1150,y=400)

thresholdBtnIcon=PhotoImage(file= r"./NewIcons/pixels.png")
thresholdBtn = Button(test, image=thresholdBtnIcon, fg='black',width=50,height=40, bg=rightIconColor,border=0, command=thresholdBtn)
thresholdBtn.place(x=1250,y=375)
tip.bind_widget(thresholdBtn,balloonmsg="Threshold: used to convert a grayscale image to a binary image by setting"+"\n" 
                                        +"all pixel values above a certain threshold to one intensity (e.g., white)")
changeOnHover(thresholdBtn)

edgeBtn_CannyIcon=PhotoImage(file= r"./NewIcons/hazard.png")
edgeBtn_Canny = Button(test, image=edgeBtn_CannyIcon,fg='black',width=50,height=40, bg=rightIconColor,border=0, command=edgeDetectCanny)
edgeBtn_Canny.place(x=1335,y=375)
tip.bind_widget(edgeBtn_Canny,balloonmsg="canny edge detection: use to find the all edges in image"+"\n"+"by threshold values min and max")
changeOnHover(edgeBtn_Canny)

labelMaxGray = Label(test, text="Gray-scale Values", border=0 ,width=15,height=2,bg='#EDCBD2',fg=textColor,font=("Helvetica", 8), relief= RAISED, borderwidth=0)
labelMaxGray.place(x=1105, y=440)

labelMinGray = Label(test, text="Min", border=0 ,width=5,height=2,bg='#EDCBD2',fg=textColor,font=("Helvetica", 10), relief= RAISED, borderwidth=0)
labelMinGray.place(x=1100, y=465)

labelMaxGray = Label(test, text="Max", border=0 ,width=5,height=2,bg='#EDCBD2',fg=textColor,font=("Helvetica", 10), relief= RAISED, borderwidth=0)
labelMaxGray.place(x=1100, y=490)

minInputGray = Entry(test, width=10)
minInputGray.place(x=1150,y=470)
maxInputGray = Entry(test, width=10)
maxInputGray.place(x=1150,y=495)

greyLevelIcon = PhotoImage(file= r"./NewIcons/volume-control.png")
greyLevelBtn = Button(test, image=greyLevelIcon, bg=rightIconColor, border=0,width=50,height=40, command=grayMethod)
greyLevelBtn.place(x=1250,y=470)
tip.bind_widget(greyLevelBtn,balloonmsg="Gray level slicing: is a way to highlight gray range of interest to a viewer by one of two ways"+"\n"
                                    +"and can select two value from inputs of Gray level"+"\n"
                                    +"if you dont select two values it will calculate mean and median of image and do gray level")
changeOnHover(greyLevelBtn)

labelMaxGray = Label(test, text="Frequency Filter", border=0 ,width=15,height=2,bg='#EDCBD2',fg=textColor,font=("Helvetica", 8), relief= RAISED, borderwidth=0)
labelMaxGray.place(x=1320, y=440)

frequancyFilterInput = Entry(test, width=10)
frequancyFilterInput.place(x=1325,y=470)


lowFrequencylIcon = PhotoImage(file= r"./NewIcons/icons8-minimum-value-32.png")
lowFrequencyBtn = Button(test, image=lowFrequencylIcon, bg=rightIconColor, border=0,width=50,height=40, command=lowFreq)
lowFrequencyBtn.place(x=1400,y=470)
tip.bind_widget(lowFrequencyBtn,balloonmsg="Low Pass Filter: the low pass filter only allows low frequency signals from 0Hz to its cut-off frequency")
changeOnHover(lowFrequencyBtn)

heighFrequencylIcon = PhotoImage(file= r"./NewIcons/icons8-bell-curve-32.png")
heighFrequencyBtn = Button(test, image=heighFrequencylIcon, bg=rightIconColor, border=0,width=50,height=40, command=highFreq)
heighFrequencyBtn.place(x=1470,y=470)
tip.bind_widget(heighFrequencyBtn,balloonmsg="High-pass filter: task is just the opposite of a low-pass filter: to offer easy passage of a high-frequency signal and difficult passage to a low-frequency signal")
changeOnHover(heighFrequencyBtn)

searingBt1Icon = PhotoImage(file= r"./NewIcons/insert-picture-icon.png")
searingBt1 =Button(test,image=searingBt1Icon,fg='black',width=50,border=0,height=40,bg=leftIconColor, command=Sharping1)
searingBt1.place(x=1100,y=580)
tip.bind_widget(searingBt1,balloonmsg="Low sharping: use to do low sharping on image"+"\n"+"Must chooes sharping from filters to can use it")
changeOnHover(searingBt1)

searingBt2Icon = PhotoImage(file= r"./NewIcons/photo (2).png")
searingBt2 =Button(test,image=searingBt2Icon,fg='black',bg=leftIconColor,border=0,width=50,height=40, command=Sharping2)
searingBt2.place(x=1220,y=580)
tip.bind_widget(searingBt2,balloonmsg="Medium sharping: use to do medium sharping on image"+"\n"+"Must chooes sharping from filters to can use it")
changeOnHover(searingBt2)

searingBt3Icon = PhotoImage(file= r"./NewIcons/image.png")
searingBt3 =Button(test,image=searingBt3Icon,fg='black',bg=leftIconColor,border=0,width=50,height=40, command=Sharping3)
searingBt3.place(x=1340,y=580)
tip.bind_widget(searingBt3,balloonmsg="High sharping: use to do high sharping on image"+"\n"+"Must chooes sharping from filters to can use it")
changeOnHover(searingBt3)

logTransformIcon = PhotoImage(file= r"./NewIcons/logarithm.png")
logTransform = Button(test, image=logTransformIcon,bg=rightIconColor, border=0 ,width=50,height=40, command=logBtn)
logTransform.place(x=1420,y=375)
tip.bind_widget(logTransform,balloonmsg="Log Filter: use to map a narrow range of dark input values into a wider range of output values")
changeOnHover(logTransform)

test.mainloop()