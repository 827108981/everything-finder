; LocalFullTextSearch installer. The setup itself requests elevation once so
; it can register the optional, on-demand NTFS change-journal helper service.
#define MyAppName "本地多格式全文搜索工具"
#ifndef MyAppVersion
  #define MyAppVersion "1.6.15"
#endif
#ifndef MyAppBuildTimestamp
  #define MyAppBuildTimestamp "202608281145"
#endif
#define MyAppPublisher "本地全文搜索工具"
#define MyAppExeName "本地多格式全文搜索工具.exe"
#ifndef SourceDir
  #define SourceDir "..\dist\本地多格式全文搜索工具-{#MyAppVersion}"
#endif

[Setup]
AppId={{0B706F7C-7CE6-4C4C-91FD-2E8C676A4F1B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion} {#MyAppBuildTimestamp}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LocalFullTextSearch
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=本地多格式全文搜索工具-安装程序-{#MyAppVersion}-{#MyAppBuildTimestamp}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\{#MyAppExeName}
; The application itself is intentionally not marked requireAdministrator.
; Only the setup needs elevation to register the metadata-only helper service.

[Languages]
Name: "default"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=本安装程序将安装主程序，并注册一个后台变更监控服务。\n\n管理员权限的用途：读取 NTFS 变化记录，以便增量更新时直接定位新增、修改和删除文件，减少重复扫描目录。服务不读取文件正文、不联网，也不改变文件内容。\n\n安装时 Windows 会请求一次管理员确认；日常启动主程序不需要管理员权限。若服务不可用，程序仍可使用普通目录扫描。

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\安装程序与管理员权限说明.md"; DestDir: "{app}"; Flags: ignoreversion

; Keep a native crash dump when Windows terminates the GUI before Python can
; flush its exception hooks.  The folder is per-user and the setting is
; removed with the application; normal Python startup diagnostics remain the
; primary evidence for recoverable failures.
[Registry]
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\{#MyAppExeName}"; ValueType: expandsz; ValueName: "DumpFolder"; ValueData: "%LOCALAPPDATA%\LocalFullTextSearch\crash-dumps"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\{#MyAppExeName}"; ValueType: dword; ValueName: "DumpType"; ValueData: "2"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\{#MyAppExeName}"; ValueType: dword; ValueName: "DumpCount"; ValueData: "10"; Flags: uninsdeletevalue

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

; Do not auto-launch the GUI after installation.  The setup process is
; elevated and may be followed by Restart Manager/installer cleanup; starting
; the GUI from this process can make the application's parent disappear while
; a scan is still starting.  The user launches the installed shortcut after
; setup has fully returned instead.

[UninstallRun]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--remove-usn-service"; RunOnceId: "RemoveLocalFullTextSearchUsnService"; Flags: runhidden waituntilterminated

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  WizardForm.StatusLabel.Caption := '正在注册 NTFS 变更监控服务...';
  if (not Exec(ExpandConstant('{app}\{#MyAppExeName}'), '--install-usn-service',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode)) or
     (ResultCode <> 0) then
  begin
    MsgBox(
      '后台变更监控服务未能注册。主程序仍可正常使用，并会在需要时回退到普通目录扫描。' + #13#10 +
      '请确认安装程序以管理员身份运行；如问题持续，请将安装目录中的说明文件和诊断日志交给支持人员。',
      mbInformation, MB_OK);
  end;
end;
