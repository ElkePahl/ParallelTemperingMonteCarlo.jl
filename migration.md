# What to do with Mahuika shutting down

Access to the new cluster can be found at <https://ondemand.nesi.org.nz>. There are two features to navigate here(most important ones to know about):
- There's a new file management system, which is much better than the old one, and is much easier to use
- Under the Clusters menu, you can open terminal access to the shell.

# Any difference between this and Mahuika?

Yes. I found that to run scripts that use OpenBLAS, you must run the following first:
```bash
export LD_LIBRARY_PATH=/opt/nesi/CS400_centos7_bdw/LegacySystemLibs/7:$LD_LIBRARY_PATH
```
Otherwise, should be smooth sailing.