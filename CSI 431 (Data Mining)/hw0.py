#   Name: Devon James       Date: 9/10/18   Class: CSI 431      Prof: Petko
#   Homework 0              Filename - hw0.py
#------------------------------------------------------------------------------

def main():

    print("Question 1")

    #Question 1 (Install/Test Python Environment)
    def test_print():
        print("This is a test statement.\n")
    if __name__ == '__main__':
        test_print()



    #Question 2 (Print size of list/size of set)
    print("Question 2")
    def list_set_length():
        #Create items_list
        items_list = [1,2,3,4,3,2,1]
        #Create items_set
        items_set = {1,2,3,4,3,2,1}

        print("Size of items_list - " + str(len(items_list)))
        print("Size of items_set - " + str(len(items_set)) + "\n")

    list_set_length()

    #Question 3 (Comprehension)
    print("Question 3")
    def set_intersect():
        #Create comprehension that only prints numbers contained in both lists
        print([x for x in {1,2,3,4} for y in {3,4,5,6} if x==y])
        print("\n")
    set_intersect()



    #Question 4 (Tuples) **Couldn't figure it out
    print("Question 4")
    def three_tuples():
        #S = {-4,-2,1,2,5,0}
        #Create triple comprehension (i,j,k) for elements whose sum = 0
        #print([i and j for i,j in {-4,-2,1,2,5,0} if i+j==0])
        print({i and j and k for i in {-4,-2,1,2,5,0} for j in {-4,-2,1,2,5,0} for k in {-4,-2,1,2,5,0} if i+j+k==0})
        print("\n")
    three_tuples()

    #Question 5 (Dictionaries) **Couldn't figure it out
    print("Question 5")
    def dict_init():
        #Initialize dictionary
        mydict = {'Neo':'Keanu', 'Morpheus':'Laurence', 'Trinity':'Carrie-Anne'}
    dict_init()

    #def dict_find():
        #


    #Question 6 (File Reading)
    print("\nQuestion 6")
    def file_line_count():
        fileName = "stories.txt"
        #Open file
        readFile = open(fileName, "r") #r for reading file

        #Read file
        #print(writeFile.read())
        currentLine = readFile.readline()
        count = 0

        while currentLine != "":
            count = count + 1
            currentLine = readFile.readline()
        print("There are " + str(count) + " lines of text within this file.\n")

        #Close file
        readFile.close()

    file_line_count()

    #Question 7 (Mini Search Engine) **Couldn't figure it out
    print("Question 7")
    #def make_inverse_index(strlist):
        #Return dictionary/inverseIndex

    #make_inverse_index(strlist)

    #def or_search(inverseIndex, query):
        #

    #or_search(inverseIndex, query)

    #def and_search(inverseIndex, query):
        #

    #and_search(inverseIndex, query)


main()
