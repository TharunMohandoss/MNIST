from PIL import Image
import os
from utils.dirs import create_dirs
import cv2
import numpy as np


def multi_epoch_viz(base_folder,epochs,rows,repeat_folder,actual_folder):
	html_text = ''		
	html_text += "<html><head></head><body><table border=\"1\"><tr><th>Sl</th><th>"+str(repeat_folder)+"</th>"
	for i in range(epochs):
		html_text += ("<th>"+str(i)+"</th>")
	html_text += ('</tr>')

	for i in range(rows):
		html_text += '<tr>'

		html_text += '<td>' + str(i)+ '</td>'

		html_text += '<td><img src="./0/'+str(repeat_folder)+"/"+str(i)+'.jpg" alt="Girl in a jacket" style="width:'+str(256)+'px;height:'+str(256)+'px;"></img></td>'

		for k in range(epochs):
			html_text += '<td><img src="./'+str(k)+'/'+str(actual_foler)+"/"+str(i)+'.jpg" alt="Girl in a jacket" style="width:'+str(256)+'px;height:'+str(256)+'px;"></img></td>'

		html_text += '</tr>'

	html_text += '</table></body></html>'

	out_file = open(os.path.join(base_folder,'multi_epoch_viz.html'),'w')
	out_file.write(html_text)
	out_file.close()

multi_epoch_viz('/media/tharun/Data/Ubuntu/Radiant/gc2019-experiment-1/Visualizations/782158477739_10minus3/train/'
	,108,200,'bgr_true_images','bgr_generated_images')










