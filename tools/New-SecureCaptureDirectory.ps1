<#
.SYNOPSIS
Create a new local Windows capture directory with an explicit restricted ACL.
.DESCRIPTION
Uses the native .NET filesystem ACL creation API. Existing targets are rejected.
The current user, SYSTEM and local Administrators receive inherited FullControl.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This tool requires Windows filesystem ACL support.'
}
$captureDirectoryPath = [IO.Path]::GetFullPath($Path)
if ([IO.Path]::GetPathRoot($captureDirectoryPath).StartsWith('\\')) {
    throw 'This tool is qualified for local Windows volumes.'
}
if ($null -ne (Get-Item -LiteralPath $captureDirectoryPath -Force -ErrorAction SilentlyContinue)) {
    throw 'The capture directory already exists. Choose a new directory.'
}

$captureOwner = [Security.Principal.WindowsIdentity]::GetCurrent().User
$capturePrincipals = @(
    $captureOwner,
    [Security.Principal.SecurityIdentifier]::new('S-1-5-18'),
    [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
)
$captureAcl = [Security.AccessControl.DirectorySecurity]::new()
$captureAcl.SetAccessRuleProtection($true, $false)
$captureAcl.SetOwner($captureOwner)
foreach ($principal in $capturePrincipals) {
    $rule = [Security.AccessControl.FileSystemAccessRule]::new(
        $principal,
        [Security.AccessControl.FileSystemRights]::FullControl,
        ([Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit),
        [Security.AccessControl.PropagationFlags]::None,
        [Security.AccessControl.AccessControlType]::Allow
    )
    $captureAcl.AddAccessRule($rule)
}
[IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($captureDirectoryPath), $captureAcl)

$actual = Get-Acl -LiteralPath $captureDirectoryPath
$actualRules = @($actual.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
$expectedSids = @($capturePrincipals | ForEach-Object Value | Sort-Object -Unique)
$actualSids = @($actualRules | ForEach-Object { $_.IdentityReference.Value } | Sort-Object -Unique)
$differences = @(Compare-Object $expectedSids $actualSids)
$fullControl = [Security.AccessControl.FileSystemRights]::FullControl
if (-not $actual.AreAccessRulesProtected -or $differences.Count -ne 0 -or
    @($actualRules | Where-Object {
        $_.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
        ($_.FileSystemRights -band $fullControl) -ne $fullControl
    }).Count -ne 0) {
    throw 'Capture directory ACL verification failed. Do not write capture data here.'
}
[pscustomobject]@{
    path = $captureDirectoryPath
    owner = $actual.GetOwner([Security.Principal.SecurityIdentifier]).Value
    inheritanceProtected = $actual.AreAccessRulesProtected
    allowedPrincipals = $actualSids
}
