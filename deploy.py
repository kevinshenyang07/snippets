import tarfile
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--package", help="specify the package path", type=str)
parser.add_argument("--install_root", help="specify the install root", type=str)
args = parser.parse_args()

package_path = args.package
install_root = args.install_root

if not install_root.endswith("/"):
    install_root = install_root + "/"


print "executing deploy script, the package path is {0}, the install_root is {1}".format(package_path, install_root)


tar = tarfile.open(package_path, "r:gz")

print "unzipping the package file to the current directory"

tar.extractall()

for tarinfo in tar:
    #This selects all the root folder
    if tarinfo.isdir() and "/" not in tarinfo.name:
        print "copying folder {0}".format(tarinfo.name)
        os.system("hdfs dfs -put {0} {1}".format(tarinfo.name, install_root))


# These codes are the old way, which create the folder first then copy the files one by one, which is much slower
# directory_creation_command = "hdfs dfs -mkdir"
# for tarinfo in tar:
#     if tarinfo.isdir():
#     	print "creating folder {0} in HDFS".format(tarinfo.name)
#     	directory_creation_command += " {0}{1}".format(install_root,tarinfo.name)


# os.system(directory_creation_command)

# print "uploading files to corresponding folders in HDFS"

# for tarinfo in tar:
#     if tarinfo.isfile():
#     	print "upload {0} file in HDFS".format(tarinfo.name)
#     	os.system("hdfs dfs -put {0} {1}{2}".format(tarinfo.name, install_root, tarinfo.name))

print "deployment completed"

tar.close()
