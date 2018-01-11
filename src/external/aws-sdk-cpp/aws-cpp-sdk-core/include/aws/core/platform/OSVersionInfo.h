/*
  * Copyright 2010-2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.
  *
  * Licensed under the Apache License, Version 2.0 (the "License").
  * You may not use this file except in compliance with the License.
  * A copy of the License is located at
  *
  *  http://aws.amazon.com/apache2.0
  *
  * or in the "license" file accompanying this file. This file is distributed
  * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
  * express or implied. See the License for the specific language governing
  * permissions and limitations under the License.
  */

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace OSVersionInfo
{
    /**
    * computing the version string for the current running operating system.
    */
    AWS_CORE_API Aws::String ComputeOSVersionString();

    /**
    * runs a (shell) command string and returns the output; not needed on windows currently
    */
    AWS_CORE_API Aws::String GetSysCommandOutput(const char* command);

} // namespace OSVersionInfo
} // namespace Aws
