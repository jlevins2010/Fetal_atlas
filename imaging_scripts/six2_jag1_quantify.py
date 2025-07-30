from ij import IJ, ImagePlus, ImageStack
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from ij.gui import Roi
from loci.plugins import BF

import csv

max_threshold = 65535

sigma = "3"
particle_size_min = "1"
particle_size_max = "Infinity"
outPath = "/Volumes/levinsohnIM/imageJ_macros/quant_data/IGF_set_day12/"

with open('/Volumes/levinsohnIM/imageJ_macros/input_csvs/Input_values_JAG_SIX2_run2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)
    row_length = len(rows)
    rt = ResultsTable()
    row_counter = 0 
    for i in (range(row_length)[1:]):
        ### import arguments from sample sheet
        JAG_thresh = int(rows[i][3])
        JAG_channel = int(rows[i][2])
        SIX2_thresh = int(rows[i][5])
        SIX2_channel = int(rows[i][4])
        lowPlane = int(rows[i][6])
        highPlane = int(rows[i][7])
        sampleName = str(rows[i][1])
        sampleType = str(rows[i][8])
        path = rows[i][0]
        #ROIpath = str(rows[i][11])
        print(path)
        ### Open image
        image = BF.openImagePlus(path)

        imp = image[0]
        IJ.run(imp, "Gaussian Blur...", "sigma=" + sigma+" stack")
        IJ.run(imp, "Z Project...", "projection=[Max Intensity]")
        imp = IJ.getImage() # Get the Z-projected image
        channels = ChannelSplitter.split(imp);
        stack_SIX2 = channels[SIX2_channel]
        stack_JAG = channels[JAG_channel]
        ip = stack_SIX2.getProcessor()
        ip.setThreshold(SIX2_thresh, max_threshold)
        mask_imp = ImagePlus("temp", ip)
        IJ.run(mask_imp, "Analyze Particles...", "size=0-Infinity pixel show=Masks")
        temp_rt = ResultsTable.getResultsTable()
        if temp_rt is not None and temp_rt.getCounter() > 0:
              total_area = 0
              for k in range(temp_rt.getCounter()):
                    total_area += temp_rt.getValue("Area", k)

              rt.setValue("sample", row_counter, sampleName)
              rt.setValue("sampleType", row_counter, sampleType)
              rt.setValue("SIX2 Total Area", row_counter, total_area)
              rt.setValue("SIX2 Average Area", row_counter, total_area/temp_rt.getCounter())
              temp_rt.reset()

        else:
              rt.setValue("sample", row_counter, sampleName)
              rt.setValue("sampleType", row_counter, sampleType)
              rt.setValue("SIX2 Total Area", row_counter, 0)
              rt.setValue("SIX2 Average Area", row_counter, 0)
              temp_rt.reset()

        ip = stack_JAG.getProcessor()
        ip.setThreshold(JAG_thresh, max_threshold)
        mask_imp = ImagePlus("temp", ip)
        IJ.run(mask_imp, "Analyze Particles...", "size=20-Infinity pixel show=Masks")
        temp_rt = ResultsTable.getResultsTable()
        if temp_rt is not None and temp_rt.getCounter() > 0:
              total_area = 0
              for k in range(temp_rt.getCounter()):
                    total_area += temp_rt.getValue("Area", k)

              rt.setValue("sample", row_counter, sampleName)
              rt.setValue("sampleType", row_counter, sampleType)
              rt.setValue("JAG1 Total Area", row_counter, total_area)
              rt.setValue("JAG1 Average Area", row_counter, total_area/temp_rt.getCounter())
              row_counter += 1
              temp_rt.reset()

        else:
              rt.setValue("sample", row_counter, sampleName)
              rt.setValue("sampleType", row_counter, sampleType)
              rt.setValue("JAG1 Total Area", row_counter, 0)
              rt.setValue("JAG1 Average Area", row_counter, 0)
              row_counter += 1
              temp_rt.reset()        
                
    savePath = outPath+"Six2_JAG_day12_june_run2.csv"
    rt.saveAs(savePath)

        
# Close the image
print("completed!")
