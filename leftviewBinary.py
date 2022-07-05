from collections import deque

class node:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None


class leftTraversal:
    #def __init__(self):

    lev1= 0
    lev2 = 0
    def traverseleft(self, root, level):

        if not root :
            return
        sq = deque()
        sq.append(root)


        while sq:

            n= len(sq)
            temp =n#temp
            while n>0:
                n-=1
                elem = sq.popleft()
                if n+1==temp:
                    print(elem.data)
                if(elem.left):
                    sq.append(elem.left)
                if elem.right:
                   sq.append(elem.right)

       #  #lev=0
       #  if not root :
       #      return
       #
       #  #print(self.lev , level)
       #  if self.lev1<level:
       #      #print(root.data)
       #      self.lev1 = level
       #
       #  self.traverseleft(root.left,level+1)
       #
       #  print(root.data,end= " ")
       # # self.lev1=0
       #  #self.traverseright(root,1)

    def traverseright(self, root, level):
            # lev=0
        if not root:
            return

            # print(self.lev , level)
        if self.lev2 < level:
            print(root.data,end=" ")
            self.lev2 = level

        self.traverseright(root.right, level + 1)

            #print(root.data, end=" ")
        #self.traverse(root.right, level + 1)
        #print(root.data)

if __name__ =="__main__":

    root = node(1)

    root.left = node(2)
    root.right = node(3)
    root.left.right = node(5)
   # root.left.left = node(4)
    root.right.left = node(6)
    root.right.right=node(7)
    left = leftTraversal()
    left.traverseleft(root,1)
    #left.traverseright(root, 2)



