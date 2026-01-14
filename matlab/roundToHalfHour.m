function dn = roundToHalfHour(dt)
    vec = datevec(dt);
    vec(:,5) = round(vec(:,5)/30)*30;  % Round minutes
    vec(:,6) = 0;  % Zero out seconds
    dn = datenum(vec);
end