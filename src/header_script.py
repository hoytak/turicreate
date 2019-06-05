import os
from os.path import join, split, relpath, normpath, abspath, exists
import re


root_dir = abspath(os.getcwd())

copyright = """
/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
"""


def do_subdir(d):

    d = relpath(abspath(d), root_dir)

    base, name = split(d)
    
    macro_name = "TURI_%s_HPP_" % re.sub("[^a-zA-Z]", "_", d).upper()
    forward_macro_name = "TURI_%s_FORWARD_HPP_" % re.sub("[^a-zA-Z]", "_", d).upper()

    out_file = join(d, name + ".hpp")
    forward_out_file = join(d, name + "_forward.hpp")
 
    data = []

    data.append("#ifndef %s" % macro_name)
    data.append("#define %s" % macro_name)
    data.append("")

    for dirpath, dirnames, filenames in os.walk(d):

        for f in filenames:
            if f.endswith(".hpp"):
                data.append("#include <%s/%s>" % (dirpath, f))

    
    data.append("")
    data.append("#endif\n")

    if not exists(out_file):
        open(out_file, "w").write("\n".join(data))

        print("Wrote out file %s." % out_file)
    else:
        print("Skipping file %s." % out_file)

    if not exists(forward_out_file):
        open(forward_out_file, "w").write(
"""
#ifndef %s
#define %s

namespace turi { 

    // Forward declarations of classes


}

#endif
""" % (forward_macro_name, forward_macro_name))


if __name__ == "__main__":

    def is_header_dir(d):
        if re.match("^core/[a-z_]+$", d):
            return True
        if d in ["core", "python"]:
            return False

        if re.match("^[a-z_]+$", d):
            return True

        return False


    paths = (relpath(d, root_dir) for d, dl, fl in os.walk(root_dir))

    for p in paths:
        if is_header_dir(p):
            print(p)
            do_subdir(p)




