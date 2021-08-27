class CoordinateRow():
    def __init__(self,arr):

        self.row=arr

    def interlace(self, coordinate_row):
        interlaced = []
        added_self_cnt = 0
        added_other_cnt = 0

        for i in range(len(self.row) + len(coordinate_row.row)):
            #print(i)
            if added_self_cnt < len(self.row):  # more to add from self
                interlaced.append(self.row[added_self_cnt])
                added_self_cnt += 1

            if added_other_cnt < len(coordinate_row.row):  # more to add from the other one
                interlaced.append(coordinate_row.row[added_other_cnt])
                added_other_cnt += 1

        self.row = interlaced


a = CoordinateRow([(5,4),(4,5),(8,7)])
a.interlace(CoordinateRow([(6,3),(3,2),(9,6),(4,3)]))
print(a.row)