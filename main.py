# Other Python Files
from modules import models

def performValidation(myProject,
                      baseLine_show = False,
                      BoostForest_show = False,
                      LogisticReg_show = False,
                      kMeans_show = False,
                      SSSVM_show = False,
                      SpeCluster_show = False):
    if baseLine_show:
        myProject.baselineFit()
    if BoostForest_show:
        myProject.AccSqueeze_tree_toy()
        myProject.AccSqueeze_tree_main(stage = [False, False, False, True, False, False],
                                       showBest = False, verbose = 0)
        myProject.AccSqueeze_tree_last()
    if LogisticReg_show:
        myProject.AccSqueeze_linear_main(verbose = 0)
        myProject.AccSqueeze_linear_last()
    if kMeans_show:
        myProject.AccSqueeze_kMeans_main(verbose = 0)
        myProject.AccSqueeze_kMeans_last()
    if SSSVM_show:
        myProject.Explore_SSSVM_main(repeat = 50)
    if SpeCluster_show:
        #myProject.Explore_SpeCluster_main(verbose = 0)
        myProject.Explore_SpeCluster_last()

def showTestResult(myProject, show = True):
    myProject.baselineTest()
    myProject.AccSqueeze_tree_test()
    myProject.AccSqueeze_linear_test()
    myProject.AccSqueeze_kMeans_test()
    myProject.Explore_SSSVM_test()
    myProject.Explore_SpeCluster_test()


if __name__ == "__main__":
    myProject = models()
    print(myProject)
    performValidation(myProject = myProject,
                      baseLine_show = True,
                      BoostForest_show = False,
                      LogisticReg_show = False,
                      kMeans_show = False,
                      SSSVM_show = False,
                      SpeCluster_show = False)
    showTestResult(myProject = myProject, show = True)


