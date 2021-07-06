import os

# list of directories
directories = ['data', 'models', 'output'] 

def main():

    # Create directory
    for dir_name in directories:

	    try:
	        os.makedirs(dir_name)
	        print("Directory " , dir_name ,  " Created ") 
	    except FileExistsError:
	        print("Directory " , dir_name ,  " already exists")        
		    
		   
         
if __name__ == '__main__':
    main()