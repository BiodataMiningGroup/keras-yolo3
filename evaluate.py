import numpy as np


def ioU(box1, box2):
    '''
    Compute intersection over union for two boxes.

    Arguments:
    box1 -- one of the boxes given as list of tpye [ymin, xmin, ymax, xmax]
    box2 -- the other boxe given as list of tpye [ymin, xmin, ymax, xmax]

    Returns:
    The size of the intersection of box1 and box2 devided by the size of the union
     '''

    # get the overlap box
    boxO = np.concatenate((np.max([box1[:2], box2[:2]], axis=0), np.min([box1[2:], box2[2:]], axis=0))).astype(np.float)

    # compute areas of boxO box1 and box2
    areas = [np.max([box[2] - box[0], 0]) * np.max([box[3] - box[1], 0]) for box in [boxO, box1, box2]]

    # return the overlap area devided by the union
    return areas[0] / (areas[1] + areas[2] - areas[0])


def ioSD(box1, box2):
    '''
    Compute intersection over symmetric difference for two boxes.

    Arguments:
    box1 -- one of the boxes given as list of tpye [ymin, xmin, ymax, xmax]
    box2 -- the other boxe given as list of tpye [ymin, xmin, ymax, xmax]

    Returns:
    The size of the intersection of box1 and box2 devided by the size of the union
     '''

    # get the overlap box
    boxO = np.concatenate((np.max([box1[:2], box2[:2]], axis=0), np.min([box1[2:], box2[2:]], axis=0))).astype(np.float)

    # compute areas of boxO box1 and box2
    areas = [np.max([box[2] - box[0], 0]) * np.max([box[3] - box[1], 0]) for box in [boxO, box1, box2]]

    # return the overlap area devided by the union
    if (areas[1] + areas[2] - 2 * areas[0]) != 0:
        return areas[0] / (areas[1] + areas[2] - 2 * areas[0])
    else:
        return 1


def findMatchingBoxes(boxes1, boxes2, measure=ioU, thresh=0.5):
    '''
    for each box $B_1$ in boxes1 find all boxes $B_2$ in boxes2 with ioU(B_1, B_2) > ioU_thresh

    Arguments:
    boxes1 -- A list of boxes given as list of lists[[ymin, xmin, ymax, xmax], ...]. For each of this boxes their matches in boxes2 are determined
    boxes2 -- A list of boxes given as list of lists[[ymin, xmin, ymax, xmax], ...]. For each box $B_1$ in boxes1, matches of $B_1$ are searched in this list
    ioU_thresh -- The threshold  determines the minimum ioU of two boxes that is sufficient to consider two boxes matching

    Returns:
    A list of lists where the i-th list contains the ids of all boxes in boxes2 that match with the i-th box in boxes1
    '''
    return [[b_id for b_id, box2 in enumerate(boxes2) if measure(box1, box2) > thresh] for box1 in boxes1]


def countHits(detectedBoxes, groundtruthBoxes, measure=ioU, thresh=0.5):
    '''
    Count the True positve, False positive and False negative detected boxes

    Arguments:
    detectedBoxes -- The detected boxes that will be evaluated given as list of lists[[ymin, xmin, ymax, xmax], ...].
    grountruthBoxes -- the groundTruth boxes iven as list of lists[[ymin, xmin, ymax, xmax], ...].
    ioU_tresh -- The intersection over union threshold that is sufficient to consider a detected box as TP

    Results:
    The number of true positives, false positives and false negatives respectively
    '''

    matches = findMatchingBoxes(detectedBoxes, groundtruthBoxes, measure=measure, thresh=thresh)
    TP = len(list(filter(lambda x: len(x) > 0, matches)))
    FP = len(matches) - TP
    FN = len(groundtruthBoxes) - len(set([item for sublist in matches for item in sublist]))
    return TP, FP, FN


if __name__ == '__main__':

    # define ground-truth and detection
    gt = [[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1], [10, 10, 12, 13], [5, 20, 8, 22]]
    dt = [[100, 100, 200, 200], [0, 0, 1.1, 1.1], [10, 10.5, 12, 12.8]]

#    # test ioU
#    print 'ioU values for each detected box:'
#    for box1 in dt:
#        print str(box1) + ':', [(box2, ioU(box1, box2)) for box2 in gt]
#    print ''

#    # test findMatchingBoxes
#    print 'matches for each detectec box:'
#    matches = findMatchingBoxes(dt, gt)
#    for i, box1 in enumerate(dt):
#        print str(box1) + ':', [gt[m_id] for m_id in matches[i]]
#    print ''

#    # test countHits
#    print 'TP, FP, FN are', countHits(dt, gt), ', respectively.'
