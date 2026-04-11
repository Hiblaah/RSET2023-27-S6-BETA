using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Windows.Media.Control;

class MediaBridge
{
    static GlobalSystemMediaTransportControlsSessionManager manager;
    static GlobalSystemMediaTransportControlsSession currentSession;

    static async Task Main(string[] args)
    {
        manager = await GlobalSystemMediaTransportControlsSessionManager.RequestAsync();
        manager.CurrentSessionChanged += OnSessionChanged;

        AttachSession(manager.GetCurrentSession());

        // Read commands from Java via stdin
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            try
            {
                if (line == "PLAY")
                    await currentSession?.TryPlayAsync();

                else if (line == "PAUSE")
                    await currentSession?.TryPauseAsync();

                else if (line == "NEXT")
                    await currentSession?.TrySkipNextAsync();

                else if (line == "PREV")
                    await currentSession?.TrySkipPreviousAsync();

                else if (line.StartsWith("SEEK:"))
                {
                    long seconds = long.Parse(line.Substring(5));
                    await currentSession?.TryChangePlaybackPositionAsync(
                        TimeSpan.FromSeconds(seconds).Ticks);
                }

                else if (line.StartsWith("VOLUME:"))
                {
                    // Volume is system-level, handled via keypress simulation
                    // Not available through SMTC directly
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("Command error: " + e.Message);
            }
        }
    }

    static void OnSessionChanged(
        GlobalSystemMediaTransportControlsSessionManager sender,
        CurrentSessionChangedEventArgs args)
    {
        AttachSession(sender.GetCurrentSession());
    }

    static void AttachSession(GlobalSystemMediaTransportControlsSession session)
    {
        if (session == null) return;

        // Detach old
        if (currentSession != null)
        {
            currentSession.MediaPropertiesChanged -= OnMediaChanged;
            currentSession.PlaybackInfoChanged -= OnPlaybackChanged;
            currentSession.TimelinePropertiesChanged -= OnTimelineChanged;
        }

        currentSession = session;
        currentSession.MediaPropertiesChanged += OnMediaChanged;
        currentSession.PlaybackInfoChanged += OnPlaybackChanged;
        currentSession.TimelinePropertiesChanged += OnTimelineChanged;

        // Send current state immediately
        SendCurrentState();
    }

    static async void SendCurrentState()
    {
        try
        {
            if (currentSession == null)
            {
                Console.WriteLine("{\"active\":false}");
                return;
            }

            var props = await currentSession.TryGetMediaPropertiesAsync();
            var playback = currentSession.GetPlaybackInfo();
            var timeline = currentSession.GetTimelineProperties();

            bool isPlaying = playback.PlaybackStatus ==
                GlobalSystemMediaTransportControlsSessionPlaybackStatus.Playing;

            long position = (long)timeline.Position.TotalSeconds;
            long duration = (long)timeline.EndTime.TotalSeconds;

            var obj = new
            {
                active = true,
                title = props.Title ?? "",
                artist = props.Artist ?? "",
                playing = isPlaying,
                position,
                duration
            };

            string json = JsonSerializer.Serialize(obj);
            Console.WriteLine(json);
            Console.Out.Flush();
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("State error: " + e.Message);
        }
    }

    static void OnMediaChanged(
        GlobalSystemMediaTransportControlsSession session,
        MediaPropertiesChangedEventArgs args) => SendCurrentState();

    static void OnPlaybackChanged(
        GlobalSystemMediaTransportControlsSession session,
        PlaybackInfoChangedEventArgs args) => SendCurrentState();

    static void OnTimelineChanged(
        GlobalSystemMediaTransportControlsSession session,
        TimelinePropertiesChangedEventArgs args) => SendCurrentState();
}