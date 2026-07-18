# Safety Statement #

## Running User Suggestions ##

For security purposes, you are not advised to use the administrator account, such as root, to run any commands.

## File permission control ##

 *  You are advised to set the umask value of the running system to 0027 or higher on hosts (including host machines) and containers to ensure that the default maximum permission for new folders is 750 and the default maximum permission for new files is 640.
 *  It is recommended that users take security measures such as permission control for sensitive content, such as personal privacy data, business assets, source files, and files saved during operator development. For example, the permission control for installation directories and public data files is involved. For details about the permission control, see.[A-Recommended maximum number of file (folder) permission control in each scenario](#a-recommended-maximum-number-of-file-folder-permission-control-in-each-scenario)	.
 *  When an operator is running, the operator compilation file may be cached and stored in the`kernel_meta_*`In the folder, the subsequent operator invoking speed is accelerated. You can control the permission on the generated files as required.
 *  Permission control is required during installation and use. For details, see.[A-Recommended maximum number of file (folder) permission control in each scenario](#a-recommended-maximum-number-of-file-folder-permission-control-in-each-scenario)	For details about how to set the file permission, see.

## Building a Security Statement ##

When compiling the source code and installing the project, you need to compile the project by yourself. Some intermediate files are generated during the compilation. You are advised to control the permission on the intermediate files after the compilation is complete to ensure file security.

## Operation Safety Statement ##

 *  You are advised to compile the operator invoking script based on the resource status of the running environment. If the operator invoking script does not match the resource status, for example, the space used by the generated input data or benchmark calculation result exceeds the memory capacity limit, or the space used by the script to save data locally exceeds the disk space size, errors may occur and the process exits unexpectedly.
 *  When the operator runs abnormally, the process exits and an error message is printed. You are advised to locate the fault based on the error message, including setting operator synchronization and viewing log files.
 *  The operator passes the[PyTorch](https://gitee.com/ascend/pytorch)	When the mode is invoked, the running may fail due to version mismatch. For details, see.[PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)	.

## Public Network Address Statement ##

The public network address declaration contained in the project code is as follows:

|    Type    | Open source code address | File Name                              | Public IP address, URL, domain name, email address, or compressed file address                                                  | Usage Description                                                                          |
|:----------:|:------------------------:|:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| Depends on |      Not involved.       | cmake/third_party/makeself-fetch.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | Download the makeself source code from gitcode and use the compilation dependency.         |
| Depends on |      Not involved.       | cmake/third_party/protobuf.cmake       | https://github.com/protocolbuffers/protobuf/archive/v25.1.tar.gz                                                                | Download the protobuf source code from GitHub and use the compilation dependency.          |
| Depends on |      Not involved.       | cmake/third_party/json.cmake           | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip                                             | Download the JSON source code from gitcode and use the compilation dependency.             |
| Depends on |      Not involved.       | cmake/third_party/gtest.cmake          | https://github.com/google/googletest/archive/release-1.8.0.tar.gz                                                               | Download the Googletest source code from GitHub and use the compilation dependency.        |
| Depends on |      Not involved.       | cmake/third_party/secure_c.cmake       | https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz                                                    | Download the libboundscheck source code from the Gitee and use the compilation dependency. |

## Vulnerability Mechanism Description ##

[Vulnerability management](https://gitcode.com/cann/community/blob/master/security/security.md)	

## Appendixes ##

### A-Recommended maximum number of file (folder) permission control in each scenario ###

| Type                                                                        | Linux Permission Reference Max. |
| --------------------------------------------------------------------------- | ------------------------------- |
| User Home Directory                                                         | 750 (rwxr-x---)                 |
| Program files (including script files and library files)                    | 550 (r-xr-x---)                 |
| Program File Directory                                                      | 550 (r-xr-x---)                 |
| Configuration File                                                          | 640 (rw-r-----)                 |
| Configuration File Directory                                                | 750 (rwxr-x---)                 |
| Log files (recorded or archived)                                            | 440 (r--r-----)                 |
| Log File (Recording)                                                        | 640 (rw-r-----)                 |
| Log File Directory                                                          | 750 (rwxr-x---)                 |
| Debug File                                                                  | 640 (rw-r-----)                 |
| Debug File Directory                                                        | 750 (rwxr-x---)                 |
| Temporary File Directory                                                    | 750 (rwxr-x---)                 |
| Maintaining the Upgrade File Directory                                      | 770 (rwxrwx---)                 |
| Service data file                                                           | 640 (rw-r-----)                 |
| Service data file directory                                                 | 750 (rwxr-x---)                 |
| Key components, private keys, certificates, and ciphertext file directories | 700 (rwx------)                 |
| Key component, private key, certificate, and encryption ciphertext          | 600 (rw-------)                 |
| Encryption and decryption interfaces and encryption and decryption scripts  | 500 (r-x------)                 |

