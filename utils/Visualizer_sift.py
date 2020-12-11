from PIL import Image
import os
from utils.dirs import create_dirs
import cv2
import numpy as np


class Visualizer:

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.html_text = ''
        self.current_index = 0

    def reset(self):
        self.html_text = ''
        self.current_index = 0

    def Visualize(self, data_dict, order, type_list, cur_epoch, accumulate):
        top_folder = os.path.join(self.config.visualization_dir, self.mode)
        directories_to_create = [top_folder]
        top_folder = os.path.join(
            self.config.visualization_dir, self.mode, str(cur_epoch))
        directories_to_create = [top_folder]
        i = 0
        for i in range(len(order)):
            element = order[i]
            current_type = type_list[i]
            if current_type == "image":
                directories_to_create.append(os.path.join(top_folder, element))
            if current_type == "satimage":
                directories_to_create.append(os.path.join(top_folder, element))
                for i in range(len(self.config.bands_list)):
                    directories_to_create.append(os.path.join(
                        top_folder, element, self.config.bands_list[i]))
            if current_type == "imglist":
                directories_to_create.append(os.path.join(top_folder, element))
                for i in range(6):
                    directories_to_create.append(
                        os.path.join(top_folder, element, str(i)))
        create_dirs(directories_to_create)

        file = open(os.path.join(top_folder, "visualize.html"), 'w')
        starting_text = "<html><head></head><body><table border=\"1\"><tr><th>Sl</th>"
        file.write(starting_text)
        for i in range(len(order)):
            file.write("<th>"+str(order[i])+"</th>")
        file.write('</tr>')

        # add the rows
        for i in range(len(data_dict[order[0]])):
            cur_i = i + self.current_index
            self.html_text += '<tr><td>'+str(cur_i)+'</td>'
            #print("DADADA",len(data_dict[order[0]]),len(order),type(order))
            for j in range(len(order)):
                self.html_text += '<td>'
                element = order[j]
                current_type = type_list[j]
                data = data_dict[element][i]

                if current_type == "image":
                    file_name = os.path.join(
                        top_folder, element, str(cur_i)+".jpg")
                    data = (data + 1) * 127.5
                    cv2.imwrite(file_name, np.transpose(data[0:3], (1, 2, 0)))
                    string_to_write = '<img src="./'+str(element)+"/"+str(cur_i)+'.jpg" alt="Girl in a jacket" style="width:'+str(
                        data.shape[2])+'px;height:'+str(data.shape[1])+'px;"></img>'
                    self.html_text += string_to_write
                elif current_type == "satimage":
                    string_to_write = "<table border=\"1\"><tr>"
                    bands_html_list = ""
                    images_html_list = ""
                    for k in range(len(self.config.bands_list)):
                        bands_html_list += ('<td>' +
                                            self.config.bands_list[k]+'</td>')
                        file_name = os.path.join(
                            top_folder, element, self.config.bands_list[k], str(cur_i)+".jpg")
                        images_html_list += ('<td>'+'<img src="./'+str(element)+"/"+self.config.bands_list[k]+"/"+str(
                            cur_i)+'.jpg" alt="Girl in a jacket" style="width:'+str(data.shape[2])+'px;height:'+str(data.shape[1])+'px;"></img>'+'</td>')
                        cv2.imwrite(file_name, 255*data[k])
                    string_to_write += bands_html_list + "</tr><tr>"+images_html_list+"</tr></table>"
                    self.html_text += string_to_write
                elif current_type == "imglist":
                    # self.html_text+='ac'
                    string_to_write = "<table border=\"1\"><tr>"
                    bands_html_list = ""
                    images_html_list = ""
                    for k in range(6):
                        bands_html_list += ('<td>'+str(k)+'</td>')
                        file_name = os.path.join(
                            top_folder, element, str(k), str(cur_i)+".jpg")
                        images_html_list += ('<td>'+'<img src="./'+str(element)+"/"+str(k)+"/"+str(
                            cur_i)+'.jpg" alt="Girl in a jacket" style="width:'+str(data.shape[2])+'px;height:'+str(data.shape[3])+'px;"></img>'+'</td>')
                        cv2.imwrite(file_name, 255.0 *
                                    np.transpose(data[k], (1, 2, 0)))

                    string_to_write += bands_html_list + "</tr><tr>"+images_html_list+"</tr></table>"
                    self.html_text += string_to_write

                elif current_type == "numerical":
                    string_to_write = str(data)
                    self.html_text += string_to_write
                self.html_text += '</td>'
            self.html_text += '</tr>'

        self.current_index += len(data_dict[order[0]])
        file.write(self.html_text)

        ending_text = "</html>"
        file.write(ending_text)
        file.close()

        if not accumulate:
            self.reset()


def Visualize(data_dict, order, type_list, mode, directory):
    top_folder = os.path.join(directory, mode)
    directories_to_create = [top_folder]
    i = 0
    for i in range(len(order)):
        element = order[i]
        current_type = type_list[i]
        if current_type == "image":
            directories_to_create.append(os.path.join(top_folder, element))
    create_dirs(directories_to_create)

    file = open(os.path.join(top_folder, "visualize.html"), 'w')
    starting_text = "<html><head></head><body><table><tr><th>Sl</th>"
    file.write(starting_text)
    for i in range(len(order)):
        file.write("<th>"+str(order[i])+"</th>")
    file.write('</tr>')

    # add the rows
    for i in range(len(data_dict[order[0]])):
        file.write('<tr><td>'+str(i)+'</td>')
        for j in range(len(order)):
            file.write('<td>')
            element = order[j]
            current_type = type_list[j]
            data = data_dict[element][i]
            if current_type == "image":
                file_name = os.path.join(top_folder, element, str(i)+".jpg")
                cv2.imwrite(file_name, 255.0 *
                            np.transpose(data, (1, 2, 0))[..., ::-1])
                string_to_write = '<img src="./'+str(element)+"/"+str(i)+'.jpg" alt="Girl in a jacket" style="width:'+str(
                    data.shape[2])+'px;height:'+str(data.shape[1])+'px;"></img>'
                file.write(string_to_write)
            elif current_type == "numerical":
                string_to_write = str(data)
                file.write(string_to_write)

            file.write('</td>')

        file.write('</tr>')

    ending_text = "</html>"
    file.close()
