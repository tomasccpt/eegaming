#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.1),
    on abril 23, 2024, at 14:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.1'
expName = 'Motor Imagery'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'name': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Simão Francisco\\Desktop\\Miguel BCI\\Protocol\\MotorImageryExp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 720], fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('keyWelcome') is None:
        # initialise keyWelcome
        keyWelcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyWelcome',
        )
    if deviceManager.getDevice('keyOrientation2Message') is None:
        # initialise keyOrientation2Message
        keyOrientation2Message = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyOrientation2Message',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    textWelcomeMessage = visual.TextStim(win=win, name='textWelcomeMessage',
        text='Bem-Vindo ao Experimento!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank200" ---
    textBlank200 = visual.TextStim(win=win, name='textBlank200',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "StartScreen" ---
    textOrientationMessage = visual.TextStim(win=win, name='textOrientationMessage',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyWelcome = keyboard.Keyboard(deviceName='keyWelcome')
    
    # --- Initialize components for Routine "start" ---
    polygon_start = visual.Rect(
        win=win, name='polygon_start',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, 1.0000, 1.0000], fillColor=[-1.0000, 1.0000, 1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "rest" ---
    textrest = visual.TextStim(win=win, name='textrest',
        text='Descansa durante 4s',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    polygon_rest = visual.Rect(
        win=win, name='polygon_rest',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "task" ---
    polygonDE = visual.Rect(
        win=win, name='polygonDE',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "Start2Screen" ---
    textOrientationMessage2 = visual.TextStim(win=win, name='textOrientationMessage2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyOrientation2Message = keyboard.Keyboard(deviceName='keyOrientation2Message')
    
    # --- Initialize components for Routine "start" ---
    polygon_start = visual.Rect(
        win=win, name='polygon_start',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, 1.0000, 1.0000], fillColor=[-1.0000, 1.0000, 1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "rest" ---
    textrest = visual.TextStim(win=win, name='textrest',
        text='Descansa durante 4s',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    polygon_rest = visual.Rect(
        win=win, name='polygon_rest',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "task2" ---
    polygonPP = visual.Rect(
        win=win, name='polygonPP',
        width=(0.2, 0.4)[0], height=(0.2, 0.4)[1],
        ori=0.0, pos=(-0.80,-0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "GoodByeScreen" ---
    textGoodBuyMessage = visual.TextStim(win=win, name='textGoodBuyMessage',
        text='Obrigado por participar',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('WelcomeScreen.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    WelcomeScreenComponents = [textWelcomeMessage]
    for thisComponent in WelcomeScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textWelcomeMessage* updates
        
        # if textWelcomeMessage is starting this frame...
        if textWelcomeMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textWelcomeMessage.frameNStart = frameN  # exact frame index
            textWelcomeMessage.tStart = t  # local t and not account for scr refresh
            textWelcomeMessage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textWelcomeMessage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWelcomeMessage.started')
            # update status
            textWelcomeMessage.status = STARTED
            textWelcomeMessage.setAutoDraw(True)
        
        # if textWelcomeMessage is active this frame...
        if textWelcomeMessage.status == STARTED:
            # update params
            pass
        
        # if textWelcomeMessage is stopping this frame...
        if textWelcomeMessage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textWelcomeMessage.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                textWelcomeMessage.tStop = t  # not accounting for scr refresh
                textWelcomeMessage.tStopRefresh = tThisFlipGlobal  # on global time
                textWelcomeMessage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textWelcomeMessage.stopped')
                # update status
                textWelcomeMessage.status = FINISHED
                textWelcomeMessage.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('WelcomeScreen.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "blank200" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('blank200.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    blank200Components = [textBlank200]
    for thisComponent in blank200Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank200" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.2:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlank200* updates
        
        # if textBlank200 is starting this frame...
        if textBlank200.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlank200.frameNStart = frameN  # exact frame index
            textBlank200.tStart = t  # local t and not account for scr refresh
            textBlank200.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlank200, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlank200.started')
            # update status
            textBlank200.status = STARTED
            textBlank200.setAutoDraw(True)
        
        # if textBlank200 is active this frame...
        if textBlank200.status == STARTED:
            # update params
            pass
        
        # if textBlank200 is stopping this frame...
        if textBlank200.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlank200.tStartRefresh + .2-frameTolerance:
                # keep track of stop time/frame for later
                textBlank200.tStop = t  # not accounting for scr refresh
                textBlank200.tStopRefresh = tThisFlipGlobal  # on global time
                textBlank200.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank200.stopped')
                # update status
                textBlank200.status = FINISHED
                textBlank200.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank200Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank200" ---
    for thisComponent in blank200Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('blank200.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.200000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=2.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_3 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('Stimulation.xlsx'),
            seed=None, name='trials_3')
        thisExp.addLoop(trials_3)  # add the loop to the experiment
        thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        for thisTrial_3 in trials_3:
            currentLoop = trials_3
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
            if thisTrial_3 != None:
                for paramName in thisTrial_3:
                    globals()[paramName] = thisTrial_3[paramName]
            
            # --- Prepare to start Routine "StartScreen" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('StartScreen.started', globalClock.getTime(format='float'))
            textOrientationMessage.setText(instructions1)
            keyWelcome.keys = []
            keyWelcome.rt = []
            _keyWelcome_allKeys = []
            # keep track of which components have finished
            StartScreenComponents = [textOrientationMessage, keyWelcome]
            for thisComponent in StartScreenComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "StartScreen" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textOrientationMessage* updates
                
                # if textOrientationMessage is starting this frame...
                if textOrientationMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textOrientationMessage.frameNStart = frameN  # exact frame index
                    textOrientationMessage.tStart = t  # local t and not account for scr refresh
                    textOrientationMessage.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textOrientationMessage, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textOrientationMessage.started')
                    # update status
                    textOrientationMessage.status = STARTED
                    textOrientationMessage.setAutoDraw(True)
                
                # if textOrientationMessage is active this frame...
                if textOrientationMessage.status == STARTED:
                    # update params
                    pass
                
                # *keyWelcome* updates
                waitOnFlip = False
                
                # if keyWelcome is starting this frame...
                if keyWelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    keyWelcome.frameNStart = frameN  # exact frame index
                    keyWelcome.tStart = t  # local t and not account for scr refresh
                    keyWelcome.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(keyWelcome, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'keyWelcome.started')
                    # update status
                    keyWelcome.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(keyWelcome.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(keyWelcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if keyWelcome.status == STARTED and not waitOnFlip:
                    theseKeys = keyWelcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _keyWelcome_allKeys.extend(theseKeys)
                    if len(_keyWelcome_allKeys):
                        keyWelcome.keys = _keyWelcome_allKeys[-1].name  # just the last key pressed
                        keyWelcome.rt = _keyWelcome_allKeys[-1].rt
                        keyWelcome.duration = _keyWelcome_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in StartScreenComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "StartScreen" ---
            for thisComponent in StartScreenComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('StartScreen.stopped', globalClock.getTime(format='float'))
            # check responses
            if keyWelcome.keys in ['', [], None]:  # No response was made
                keyWelcome.keys = None
            trials_3.addData('keyWelcome.keys',keyWelcome.keys)
            if keyWelcome.keys != None:  # we had a response
                trials_3.addData('keyWelcome.rt', keyWelcome.rt)
                trials_3.addData('keyWelcome.duration', keyWelcome.duration)
            # the Routine "StartScreen" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "start" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('start.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            startComponents = [polygon_start]
            for thisComponent in startComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "start" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *polygon_start* updates
                
                # if polygon_start is starting this frame...
                if polygon_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon_start.frameNStart = frameN  # exact frame index
                    polygon_start.tStart = t  # local t and not account for scr refresh
                    polygon_start.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon_start, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_start.started')
                    # update status
                    polygon_start.status = STARTED
                    polygon_start.setAutoDraw(True)
                
                # if polygon_start is active this frame...
                if polygon_start.status == STARTED:
                    # update params
                    pass
                
                # if polygon_start is stopping this frame...
                if polygon_start.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon_start.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon_start.tStop = t  # not accounting for scr refresh
                        polygon_start.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon_start.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_start.stopped')
                        # update status
                        polygon_start.status = FINISHED
                        polygon_start.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in startComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "start" ---
            for thisComponent in startComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('start.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # set up handler to look after randomisation of conditions etc
            imageryTrials = data.TrialHandler(nReps=4.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('Stimulation.xlsx'),
                seed=None, name='imageryTrials')
            thisExp.addLoop(imageryTrials)  # add the loop to the experiment
            thisImageryTrial = imageryTrials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisImageryTrial.rgb)
            if thisImageryTrial != None:
                for paramName in thisImageryTrial:
                    globals()[paramName] = thisImageryTrial[paramName]
            
            for thisImageryTrial in imageryTrials:
                currentLoop = imageryTrials
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisImageryTrial.rgb)
                if thisImageryTrial != None:
                    for paramName in thisImageryTrial:
                        globals()[paramName] = thisImageryTrial[paramName]
                
                # --- Prepare to start Routine "rest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('rest.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                restComponents = [textrest, polygon_rest]
                for thisComponent in restComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "rest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *textrest* updates
                    
                    # if textrest is starting this frame...
                    if textrest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textrest.frameNStart = frameN  # exact frame index
                        textrest.tStart = t  # local t and not account for scr refresh
                        textrest.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textrest, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textrest.started')
                        # update status
                        textrest.status = STARTED
                        textrest.setAutoDraw(True)
                    
                    # if textrest is active this frame...
                    if textrest.status == STARTED:
                        # update params
                        pass
                    
                    # if textrest is stopping this frame...
                    if textrest.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > textrest.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            textrest.tStop = t  # not accounting for scr refresh
                            textrest.tStopRefresh = tThisFlipGlobal  # on global time
                            textrest.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textrest.stopped')
                            # update status
                            textrest.status = FINISHED
                            textrest.setAutoDraw(False)
                    
                    # *polygon_rest* updates
                    
                    # if polygon_rest is starting this frame...
                    if polygon_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        polygon_rest.frameNStart = frameN  # exact frame index
                        polygon_rest.tStart = t  # local t and not account for scr refresh
                        polygon_rest.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(polygon_rest, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_rest.started')
                        # update status
                        polygon_rest.status = STARTED
                        polygon_rest.setAutoDraw(True)
                    
                    # if polygon_rest is active this frame...
                    if polygon_rest.status == STARTED:
                        # update params
                        pass
                    
                    # if polygon_rest is stopping this frame...
                    if polygon_rest.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > polygon_rest.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            polygon_rest.tStop = t  # not accounting for scr refresh
                            polygon_rest.tStopRefresh = tThisFlipGlobal  # on global time
                            polygon_rest.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'polygon_rest.stopped')
                            # update status
                            polygon_rest.status = FINISHED
                            polygon_rest.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in restComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "rest" ---
                for thisComponent in restComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('rest.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
                
                # --- Prepare to start Routine "task" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('task.started', globalClock.getTime(format='float'))
                polygonDE.setFillColor([0.0000, 0.0000, 0.0000])
                polygonDE.setLineColor([0.0000, 0.0000, 0.0000])
                image.setImage(individual_punho)
                # keep track of which components have finished
                taskComponents = [polygonDE, image]
                for thisComponent in taskComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "task" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *polygonDE* updates
                    
                    # if polygonDE is starting this frame...
                    if polygonDE.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        polygonDE.frameNStart = frameN  # exact frame index
                        polygonDE.tStart = t  # local t and not account for scr refresh
                        polygonDE.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(polygonDE, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygonDE.started')
                        # update status
                        polygonDE.status = STARTED
                        polygonDE.setAutoDraw(True)
                    
                    # if polygonDE is active this frame...
                    if polygonDE.status == STARTED:
                        # update params
                        pass
                    
                    # if polygonDE is stopping this frame...
                    if polygonDE.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > polygonDE.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            polygonDE.tStop = t  # not accounting for scr refresh
                            polygonDE.tStopRefresh = tThisFlipGlobal  # on global time
                            polygonDE.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'polygonDE.stopped')
                            # update status
                            polygonDE.status = FINISHED
                            polygonDE.setAutoDraw(False)
                    # Run 'Each Frame' code from codeDE
                    # Altera a cor do polígono com base em uma variável
                    if individual_punho == "Imagens/right.png":
                        polygonDE.fillColor = (-0.35, -0.35, -0.35)
                    
                    # *image* updates
                    
                    # if image is starting this frame...
                    if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image.frameNStart = frameN  # exact frame index
                        image.tStart = t  # local t and not account for scr refresh
                        image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.started')
                        # update status
                        image.status = STARTED
                        image.setAutoDraw(True)
                    
                    # if image is active this frame...
                    if image.status == STARTED:
                        # update params
                        pass
                    
                    # if image is stopping this frame...
                    if image.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            image.tStop = t  # not accounting for scr refresh
                            image.tStopRefresh = tThisFlipGlobal  # on global time
                            image.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image.stopped')
                            # update status
                            image.status = FINISHED
                            image.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in taskComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "task" ---
                for thisComponent in taskComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('task.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 4.0 repeats of 'imageryTrials'
            
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_3'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_4 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('Stimulation.xlsx'),
            seed=None, name='trials_4')
        thisExp.addLoop(trials_4)  # add the loop to the experiment
        thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        for thisTrial_4 in trials_4:
            currentLoop = trials_4
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
            if thisTrial_4 != None:
                for paramName in thisTrial_4:
                    globals()[paramName] = thisTrial_4[paramName]
            
            # --- Prepare to start Routine "Start2Screen" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Start2Screen.started', globalClock.getTime(format='float'))
            textOrientationMessage2.setText(instructions2)
            keyOrientation2Message.keys = []
            keyOrientation2Message.rt = []
            _keyOrientation2Message_allKeys = []
            # keep track of which components have finished
            Start2ScreenComponents = [textOrientationMessage2, keyOrientation2Message]
            for thisComponent in Start2ScreenComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Start2Screen" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textOrientationMessage2* updates
                
                # if textOrientationMessage2 is starting this frame...
                if textOrientationMessage2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textOrientationMessage2.frameNStart = frameN  # exact frame index
                    textOrientationMessage2.tStart = t  # local t and not account for scr refresh
                    textOrientationMessage2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textOrientationMessage2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textOrientationMessage2.started')
                    # update status
                    textOrientationMessage2.status = STARTED
                    textOrientationMessage2.setAutoDraw(True)
                
                # if textOrientationMessage2 is active this frame...
                if textOrientationMessage2.status == STARTED:
                    # update params
                    pass
                
                # *keyOrientation2Message* updates
                waitOnFlip = False
                
                # if keyOrientation2Message is starting this frame...
                if keyOrientation2Message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    keyOrientation2Message.frameNStart = frameN  # exact frame index
                    keyOrientation2Message.tStart = t  # local t and not account for scr refresh
                    keyOrientation2Message.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(keyOrientation2Message, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'keyOrientation2Message.started')
                    # update status
                    keyOrientation2Message.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(keyOrientation2Message.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(keyOrientation2Message.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if keyOrientation2Message.status == STARTED and not waitOnFlip:
                    theseKeys = keyOrientation2Message.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _keyOrientation2Message_allKeys.extend(theseKeys)
                    if len(_keyOrientation2Message_allKeys):
                        keyOrientation2Message.keys = _keyOrientation2Message_allKeys[-1].name  # just the last key pressed
                        keyOrientation2Message.rt = _keyOrientation2Message_allKeys[-1].rt
                        keyOrientation2Message.duration = _keyOrientation2Message_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Start2ScreenComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Start2Screen" ---
            for thisComponent in Start2ScreenComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Start2Screen.stopped', globalClock.getTime(format='float'))
            # check responses
            if keyOrientation2Message.keys in ['', [], None]:  # No response was made
                keyOrientation2Message.keys = None
            trials_4.addData('keyOrientation2Message.keys',keyOrientation2Message.keys)
            if keyOrientation2Message.keys != None:  # we had a response
                trials_4.addData('keyOrientation2Message.rt', keyOrientation2Message.rt)
                trials_4.addData('keyOrientation2Message.duration', keyOrientation2Message.duration)
            # the Routine "Start2Screen" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "start" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('start.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            startComponents = [polygon_start]
            for thisComponent in startComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "start" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *polygon_start* updates
                
                # if polygon_start is starting this frame...
                if polygon_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon_start.frameNStart = frameN  # exact frame index
                    polygon_start.tStart = t  # local t and not account for scr refresh
                    polygon_start.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon_start, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_start.started')
                    # update status
                    polygon_start.status = STARTED
                    polygon_start.setAutoDraw(True)
                
                # if polygon_start is active this frame...
                if polygon_start.status == STARTED:
                    # update params
                    pass
                
                # if polygon_start is stopping this frame...
                if polygon_start.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon_start.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon_start.tStop = t  # not accounting for scr refresh
                        polygon_start.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon_start.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_start.stopped')
                        # update status
                        polygon_start.status = FINISHED
                        polygon_start.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in startComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "start" ---
            for thisComponent in startComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('start.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # set up handler to look after randomisation of conditions etc
            imageryTrials2 = data.TrialHandler(nReps=4.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('Stimulation.xlsx'),
                seed=None, name='imageryTrials2')
            thisExp.addLoop(imageryTrials2)  # add the loop to the experiment
            thisImageryTrials2 = imageryTrials2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisImageryTrials2.rgb)
            if thisImageryTrials2 != None:
                for paramName in thisImageryTrials2:
                    globals()[paramName] = thisImageryTrials2[paramName]
            
            for thisImageryTrials2 in imageryTrials2:
                currentLoop = imageryTrials2
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisImageryTrials2.rgb)
                if thisImageryTrials2 != None:
                    for paramName in thisImageryTrials2:
                        globals()[paramName] = thisImageryTrials2[paramName]
                
                # --- Prepare to start Routine "rest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('rest.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                restComponents = [textrest, polygon_rest]
                for thisComponent in restComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "rest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *textrest* updates
                    
                    # if textrest is starting this frame...
                    if textrest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textrest.frameNStart = frameN  # exact frame index
                        textrest.tStart = t  # local t and not account for scr refresh
                        textrest.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textrest, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textrest.started')
                        # update status
                        textrest.status = STARTED
                        textrest.setAutoDraw(True)
                    
                    # if textrest is active this frame...
                    if textrest.status == STARTED:
                        # update params
                        pass
                    
                    # if textrest is stopping this frame...
                    if textrest.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > textrest.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            textrest.tStop = t  # not accounting for scr refresh
                            textrest.tStopRefresh = tThisFlipGlobal  # on global time
                            textrest.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textrest.stopped')
                            # update status
                            textrest.status = FINISHED
                            textrest.setAutoDraw(False)
                    
                    # *polygon_rest* updates
                    
                    # if polygon_rest is starting this frame...
                    if polygon_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        polygon_rest.frameNStart = frameN  # exact frame index
                        polygon_rest.tStart = t  # local t and not account for scr refresh
                        polygon_rest.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(polygon_rest, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_rest.started')
                        # update status
                        polygon_rest.status = STARTED
                        polygon_rest.setAutoDraw(True)
                    
                    # if polygon_rest is active this frame...
                    if polygon_rest.status == STARTED:
                        # update params
                        pass
                    
                    # if polygon_rest is stopping this frame...
                    if polygon_rest.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > polygon_rest.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            polygon_rest.tStop = t  # not accounting for scr refresh
                            polygon_rest.tStopRefresh = tThisFlipGlobal  # on global time
                            polygon_rest.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'polygon_rest.stopped')
                            # update status
                            polygon_rest.status = FINISHED
                            polygon_rest.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in restComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "rest" ---
                for thisComponent in restComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('rest.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
                
                # --- Prepare to start Routine "task2" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('task2.started', globalClock.getTime(format='float'))
                polygonPP.setFillColor([-0.5500, -0.5500, -0.5500])
                polygonPP.setLineColor([-0.5500, -0.5500, -0.5500])
                image_2.setImage(ambos_punhos_pes)
                # keep track of which components have finished
                task2Components = [polygonPP, image_2]
                for thisComponent in task2Components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "task2" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *polygonPP* updates
                    
                    # if polygonPP is starting this frame...
                    if polygonPP.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        polygonPP.frameNStart = frameN  # exact frame index
                        polygonPP.tStart = t  # local t and not account for scr refresh
                        polygonPP.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(polygonPP, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygonPP.started')
                        # update status
                        polygonPP.status = STARTED
                        polygonPP.setAutoDraw(True)
                    
                    # if polygonPP is active this frame...
                    if polygonPP.status == STARTED:
                        # update params
                        pass
                    
                    # if polygonPP is stopping this frame...
                    if polygonPP.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > polygonPP.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            polygonPP.tStop = t  # not accounting for scr refresh
                            polygonPP.tStopRefresh = tThisFlipGlobal  # on global time
                            polygonPP.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'polygonPP.stopped')
                            # update status
                            polygonPP.status = FINISHED
                            polygonPP.setAutoDraw(False)
                    # Run 'Each Frame' code from code
                    # Altera a cor do polígono com base em uma variável
                    if ambos_punhos_pes == "Imagens/up.png":
                        polygonPP.fillColor = (-0.9, -0.9, -0.9)
                    
                    # *image_2* updates
                    
                    # if image_2 is starting this frame...
                    if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_2.frameNStart = frameN  # exact frame index
                        image_2.tStart = t  # local t and not account for scr refresh
                        image_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.started')
                        # update status
                        image_2.status = STARTED
                        image_2.setAutoDraw(True)
                    
                    # if image_2 is active this frame...
                    if image_2.status == STARTED:
                        # update params
                        pass
                    
                    # if image_2 is stopping this frame...
                    if image_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_2.tStartRefresh + 4-frameTolerance:
                            # keep track of stop time/frame for later
                            image_2.tStop = t  # not accounting for scr refresh
                            image_2.tStopRefresh = tThisFlipGlobal  # on global time
                            image_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_2.stopped')
                            # update status
                            image_2.status = FINISHED
                            image_2.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in task2Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "task2" ---
                for thisComponent in task2Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('task2.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 4.0 repeats of 'imageryTrials2'
            
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_4'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 2.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "GoodByeScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('GoodByeScreen.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    GoodByeScreenComponents = [textGoodBuyMessage]
    for thisComponent in GoodByeScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "GoodByeScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textGoodBuyMessage* updates
        
        # if textGoodBuyMessage is starting this frame...
        if textGoodBuyMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textGoodBuyMessage.frameNStart = frameN  # exact frame index
            textGoodBuyMessage.tStart = t  # local t and not account for scr refresh
            textGoodBuyMessage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textGoodBuyMessage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textGoodBuyMessage.started')
            # update status
            textGoodBuyMessage.status = STARTED
            textGoodBuyMessage.setAutoDraw(True)
        
        # if textGoodBuyMessage is active this frame...
        if textGoodBuyMessage.status == STARTED:
            # update params
            pass
        
        # if textGoodBuyMessage is stopping this frame...
        if textGoodBuyMessage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textGoodBuyMessage.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                textGoodBuyMessage.tStop = t  # not accounting for scr refresh
                textGoodBuyMessage.tStopRefresh = tThisFlipGlobal  # on global time
                textGoodBuyMessage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textGoodBuyMessage.stopped')
                # update status
                textGoodBuyMessage.status = FINISHED
                textGoodBuyMessage.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in GoodByeScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "GoodByeScreen" ---
    for thisComponent in GoodByeScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('GoodByeScreen.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
